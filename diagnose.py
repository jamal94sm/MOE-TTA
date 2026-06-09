"""
diagnose4.py — Fine-grained collapse tracker.
Prints every 20 batches and tracks:
  - Top-class takeover trajectory
  - Entropy percentiles (not just mean)
  - Shared vs domain expert contribution magnitude
  - Correct vs wrong sample entropy distributions
  - Global cosine LR schedule (matches main.py)

Usage:
  python diagnose4.py --data_dir ./data/ImageNet-C --severity 5 --entropy_floor 0.05
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

from config import get_cfg
from datasets import get_domain_sequence
from model import build_model
from fdd import FrequencyDomainDiscriminator


def cosine_lr(base_lr, step, total_steps):
    """Compute LR at a given step using cosine annealing."""
    if total_steps <= 0:
        return base_lr
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))


def make_optimizer(model, cfg, current_lr):
    """Create Adam optimizer for currently trainable params at given LR."""
    params = model.get_trainable_params()
    if not params:
        return None
    return torch.optim.Adam(params, lr=current_lr,
                            betas=(0.9, 0.999),
                            weight_decay=cfg.weight_decay)


def main():
    cfg = get_cfg()
    device = cfg.device
    domain_sequence = get_domain_sequence(cfg)
    total_batches = sum(len(loader) for _, loader in domain_sequence)

    model = build_model(cfg)
    fdd = FrequencyDomainDiscriminator(
        freq_radius=cfg.fdd_freq_radius,
        threshold=cfg.fdd_threshold,
        shrinkage=cfg.fdd_shrinkage,
        init_var=cfg.fdd_init_var,
        diagonal=cfg.fdd_diagonal,
        device=device,
    )

    optimizer = None
    current_fdd_domain = -1
    global_step = 0

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # save initial state
    init_shared_state = {}
    for i, em in enumerate(model.expert_modules):
        for name, p in em.shared_moe.named_parameters():
            init_shared_state[f"{i}.{name}"] = p.data.clone()

    print(f"\n{'='*90}")
    print(f"  FINE-GRAINED COLLAPSE TRACKER")
    print(f"  Entropy threshold κ = {cfg.entropy_threshold:.4f}")
    print(f"  Entropy floor       = {cfg.entropy_floor}")
    print(f"  Stochastic restore  = {cfg.stochastic_restore}")
    print(f"  Scheduler: Cosine over {total_batches} total batches "
          f"(LR: {cfg.lr} → 0)")
    print(f"{'='*90}")

    hdr = (f"  {'batch':>5} │ {'err%':>5} │ {'H_mean':>6} │ {'H_p10':>5} │ "
           f"{'H_p50':>5} │ {'H_p90':>5} │ {'filt%':>5} │ {'uniq':>4} │ "
           f"{'top_cls':>7} │ {'top%':>4} │ {'H_correct':>9} │ {'H_wrong':>7} │ "
           f"{'shrd_norm':>9} │ {'dom_norm':>8} │ {'LR':>8}")

    for seg_idx, (domain_name, loader) in enumerate(domain_sequence):
        n_batches = len(loader)
        all_errors = []
        top_class_history = []
        updates_applied = 0
        updates_skipped = 0

        print(f"\n{'─'*90}")
        print(f"  [{seg_idx+1}/{len(domain_sequence)}] {domain_name} "
              f"({len(loader.dataset)} samples, {n_batches} batches)")
        print(f"{'─'*90}")
        print(hdr)
        print(f"  {'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─"
              f"{'─'*5}─┼─{'─'*4}─┼─{'─'*7}─┼─{'─'*4}─┼─{'─'*9}─┼─{'─'*7}─┼─"
              f"{'─'*9}─┼─{'─'*8}─┼─{'─'*8}")

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            B = images.shape[0]

            # ─── FDD ───
            raw_images = images * imagenet_std + imagenet_mean
            fdd_domain_id, is_new = fdd.detect_domain(raw_images)

            if is_new:
                expert_id = model.expand_domain()
                model.set_active_domain(expert_id)
                current_fdd_domain = fdd_domain_id
                current_lr = cosine_lr(cfg.lr, global_step, total_batches)
                optimizer = make_optimizer(model, cfg, current_lr)
                print(f"  >>> FDD: New domain {fdd_domain_id} → expert {expert_id}")
            elif fdd_domain_id != current_fdd_domain:
                model.set_active_domain(fdd_domain_id)
                current_fdd_domain = fdd_domain_id
                current_lr = cosine_lr(cfg.lr, global_step, total_batches)
                optimizer = make_optimizer(model, cfg, current_lr)
                if batch_idx < 3:
                    print(f"  >>> FDD: Switched to domain {fdd_domain_id}")

            # ─── Update LR based on global step ───
            current_lr = cosine_lr(cfg.lr, global_step, total_batches)
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # ─── Forward ───
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            preds = logits.argmax(dim=-1)

            # ─── Detailed metrics ───
            err = 100.0 * (1 - (preds == labels).float().mean().item())
            all_errors.append(err)

            H_sorted = entropy.sort().values
            H_p10 = H_sorted[int(0.1 * B)].item()
            H_p50 = H_sorted[int(0.5 * B)].item()
            H_p90 = H_sorted[int(0.9 * B)].item()

            correct_mask = (preds == labels)
            H_correct = entropy[correct_mask].mean().item() if correct_mask.any() else -1
            H_wrong = entropy[~correct_mask].mean().item() if (~correct_mask).any() else -1

            pred_counts = Counter(preds.cpu().tolist())
            top_pred, top_count = pred_counts.most_common(1)[0]
            top_pct = 100 * top_count / B
            unique_preds = len(pred_counts)
            top_class_history.append((top_pred, top_pct))

            # expert param norms (block 0 as proxy)
            shared_norm = 0.0
            domain_norm = 0.0
            with torch.no_grad():
                em = model.expert_modules[0]
                shared_norm = sum(p.data.norm().item()
                                  for p in em.shared_moe.parameters()) / \
                              sum(1 for _ in em.shared_moe.parameters())
                if em.active_domain >= 0 and em.active_domain < len(em.domain_moes):
                    domain_norm = sum(p.data.norm().item()
                                      for p in em.domain_moes[em.active_domain].parameters()) / \
                                  sum(1 for _ in em.domain_moes[em.active_domain].parameters())

            # ─── Filter + Update ───
            mask = entropy < cfg.entropy_threshold
            if cfg.entropy_floor > 0:
                mask = mask & (entropy > cfg.entropy_floor)
            pass_rate = mask.float().mean().item() * 100

            if mask.sum() > 0 and optimizer is not None:
                loss = entropy[mask].mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates_applied += 1

                if cfg.stochastic_restore > 0:
                    with torch.no_grad():
                        for i, em_r in enumerate(model.expert_modules):
                            for name, p in em_r.shared_moe.named_parameters():
                                key = f"{i}.{name}"
                                if key in init_shared_state:
                                    rst = torch.rand_like(p) < cfg.stochastic_restore
                                    p.data = torch.where(rst,
                                                         init_shared_state[key],
                                                         p.data)
            else:
                updates_skipped += 1

            # ─── Print every 20 batches, first 10, and last batch ───
            if batch_idx < 10 or batch_idx % 20 == 0 or batch_idx == n_batches - 1:
                h_c_str = f"{H_correct:.3f}" if H_correct >= 0 else "  N/A"
                h_w_str = f"{H_wrong:.3f}" if H_wrong >= 0 else "  N/A"
                print(f"  {batch_idx:5d} │ {err:5.1f} │ {entropy.mean().item():6.3f} │ "
                      f"{H_p10:5.3f} │ {H_p50:5.3f} │ {H_p90:5.3f} │ "
                      f"{pass_rate:5.1f} │ {unique_preds:4d} │ "
                      f"cls{top_pred:>4d} │ {top_pct:4.0f} │ {h_c_str:>9s} │ "
                      f"{h_w_str:>7s} │ {shared_norm:9.4f} │ {domain_norm:8.4f} │ "
                      f"{current_lr:8.2e}")

            global_step += 1

        # ─── Domain summary ───
        avg_err = np.mean(all_errors)
        q = len(all_errors) // 4
        first_q = np.mean(all_errors[:q]) if q > 0 else avg_err
        last_q = np.mean(all_errors[-q:]) if q > 0 else avg_err
        trend = last_q - first_q

        takeover_batch = None
        for b_idx, (cls, pct) in enumerate(top_class_history):
            if pct > 50:
                takeover_batch = b_idx
                break

        lr_now = cosine_lr(cfg.lr, global_step, total_batches)
        print(f"\n  ┌── SUMMARY: {domain_name}")
        print(f"  │ Avg Error: {avg_err:.1f}%  "
              f"(first25%={first_q:.1f}% → last25%={last_q:.1f}%, Δ={trend:+.1f}%)")
        print(f"  │ Updates: {updates_applied} applied, {updates_skipped} skipped")
        print(f"  │ LR: {lr_now:.2e}  Step: {global_step}/{total_batches}")
        if takeover_batch is not None:
            cls, pct = top_class_history[takeover_batch]
            print(f"  │ ⚠ TAKEOVER: cls{cls} exceeded 50% at batch {takeover_batch}")
        else:
            print(f"  │ ✓ No single-class takeover")
        if trend > 10:
            print(f"  │ ⚠ DEGRADING within domain")
        print(f"  └{'─'*70}")

    print(f"\n{'='*90}")
    print(f"  DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
