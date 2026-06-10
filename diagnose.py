"""
diagnose4.py — Fine-grained collapse tracker.
Prints every 20 batches and tracks:
  - Top-class takeover trajectory
  - Entropy percentiles (not just mean)
  - Shared vs domain expert contribution magnitude
  - Correct vs wrong sample entropy distributions
  - Constant LR with persistent optimizer (preserves Adam momentum)

Usage:
  python diagnose4.py --data_dir ./data/ImageNet-C --severity 5 --entropy_floor 0.05
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

from config import get_cfg
from datasets import get_domain_sequence
from model import build_model
from fdd import FrequencyDomainDiscriminator


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
    print(f"  Diversity lambda    = {cfg.div_lambda}")
    print(f"  Optimizer: Constant LR={cfg.lr}, weight_decay={cfg.weight_decay}")
    print(f"  Total batches: {total_batches}")
    print(f"{'='*90}")

    # ─── Evaluate frozen backbone (source baseline) ───────────────────
    eval_bb = getattr(cfg, 'eval_backbone', False)
    baseline_errors = {}
    if eval_bb:
        print(f"\n  Evaluating frozen backbone on each domain...")
        model.eval()
        with torch.no_grad():
            for domain_name, loader in domain_sequence:
                correct = 0
                total = 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    # forward through backbone only (no experts)
                    logits = model.backbone(images)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
                err = 100.0 * (1 - correct / total)
                baseline_errors[domain_name] = err
                print(f"    {domain_name:25s} → {err:.1f}%")
        print(f"  Backbone mean error: "
              f"{np.mean(list(baseline_errors.values())):.1f}%\n")

    tta_errors = {}  # domain_name → avg TTA error

    hdr = (f"  {'batch':>5} │ {'err%':>5} │ {'H_mean':>6} │ {'H_p10':>5} │ "
           f"{'H_p50':>5} │ {'H_p90':>5} │ {'filt%':>5} │ {'uniq':>4} │ "
           f"{'top_cls':>7} │ {'top%':>4} │ {'H_correct':>9} │ {'H_wrong':>7} │ "
           f"{'shrd_norm':>9} │ {'dom_norm':>8}")

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
              f"{'─'*9}─┼─{'─'*8}")

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
                print(f"  >>> FDD: New domain {fdd_domain_id} → expert {expert_id}")

                if optimizer is None:
                    # first domain — create optimizer with all trainable params
                    optimizer = torch.optim.Adam(
                        model.get_trainable_params(),
                        lr=cfg.lr, betas=(0.9, 0.999),
                        weight_decay=cfg.weight_decay)
                else:
                    # add new domain expert params (preserves momentum)
                    new_domain_params = []
                    for em in model.expert_modules:
                        if expert_id < len(em.domain_moes):
                            new_domain_params.extend(
                                em.domain_moes[expert_id].parameters())
                    if new_domain_params:
                        optimizer.add_param_group({
                            'params': new_domain_params,
                            'lr': cfg.lr,
                            'betas': (0.9, 0.999),
                            'weight_decay': cfg.weight_decay,
                        })

            elif fdd_domain_id != current_fdd_domain:
                # known domain — just switch, optimizer already has params
                model.set_active_domain(fdd_domain_id)
                current_fdd_domain = fdd_domain_id
                if batch_idx < 3:
                    print(f"  >>> FDD: Switched to domain {fdd_domain_id}")

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
                # per-sample entropy (filtered)
                ent_loss = entropy[mask].mean()

                # batch diversity regularizer (all samples)
                if cfg.div_lambda > 0:
                    batch_mean_prob = probs.mean(dim=0)
                    log_C = math.log(probs.shape[-1])
                    div_loss = (batch_mean_prob * torch.log(
                        batch_mean_prob + 1e-8)).sum() + log_C
                    loss = ent_loss + cfg.div_lambda * div_loss
                else:
                    loss = ent_loss

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
                      f"{h_w_str:>7s} │ {shared_norm:9.4f} │ {domain_norm:8.4f}")

            global_step += 1

        # ─── Domain summary ───
        avg_err = np.mean(all_errors)
        tta_errors[domain_name] = avg_err
        q = len(all_errors) // 4
        first_q = np.mean(all_errors[:q]) if q > 0 else avg_err
        last_q = np.mean(all_errors[-q:]) if q > 0 else avg_err
        trend = last_q - first_q

        takeover_batch = None
        for b_idx, (cls, pct) in enumerate(top_class_history):
            if pct > 50:
                takeover_batch = b_idx
                break

        baseline_err = baseline_errors.get(domain_name, None)
        print(f"\n  ┌── SUMMARY: {domain_name}")
        if baseline_err is not None:
            improvement = baseline_err - avg_err
            arrow = "↓" if improvement > 0 else "↑"
            print(f"  │ Backbone Error:  {baseline_err:.1f}%")
            print(f"  │ TTA Error:       {avg_err:.1f}%  "
                  f"(first25%={first_q:.1f}% → last25%={last_q:.1f}%, Δ={trend:+.1f}%)")
            print(f"  │ Improvement:     {arrow} {abs(improvement):.1f}%"
                  f"{'  ⚠ TTA HURTS' if improvement < -1 else ''}")
        else:
            print(f"  │ Avg Error: {avg_err:.1f}%  "
                  f"(first25%={first_q:.1f}% → last25%={last_q:.1f}%, Δ={trend:+.1f}%)")
        print(f"  │ Updates: {updates_applied} applied, {updates_skipped} skipped")
        print(f"  │ Step: {global_step}/{total_batches}")
        if takeover_batch is not None:
            cls, pct = top_class_history[takeover_batch]
            print(f"  │ ⚠ TAKEOVER: cls{cls} exceeded 50% at batch {takeover_batch}")
        else:
            print(f"  │ ✓ No single-class takeover")
        if trend > 10:
            print(f"  │ ⚠ DEGRADING within domain")
        print(f"  └{'─'*70}")

    # ─── Final comparison table ──────────────────────────────────────────
    print(f"\n{'='*90}")
    if baseline_errors:
        print(f"  FINAL COMPARISON: Backbone vs TTA")
        print(f"{'='*90}")
        print(f"  {'Domain':<25} {'Backbone':>10} {'TTA':>10} {'Improv.':>10}")
        print(f"  {'─'*25}─{'─'*10}─{'─'*10}─{'─'*10}")

        for domain_name, _ in domain_sequence:
            b_err = baseline_errors.get(domain_name, 0)
            t_err = tta_errors.get(domain_name, 0)
            imp = b_err - t_err
            arrow = "↓" if imp > 0 else "↑"
            marker = " ⚠" if imp < -1 else ""
            print(f"  {domain_name:<25} {b_err:>9.1f}% {t_err:>9.1f}% "
                  f"{arrow} {abs(imp):>7.1f}%{marker}")

        backbone_mean = np.mean(list(baseline_errors.values()))
        tta_mean = np.mean(list(tta_errors.values()))
        mean_imp = backbone_mean - tta_mean
        print(f"  {'─'*25}─{'─'*10}─{'─'*10}─{'─'*10}")
        print(f"  {'MEAN':<25} {backbone_mean:>9.1f}% {tta_mean:>9.1f}% "
              f"{'↓' if mean_imp > 0 else '↑'} {abs(mean_imp):>7.1f}%")
    else:
        print(f"  FINAL RESULTS: TTA Adaptation Summary")
        print(f"{'='*90}")
        print(f"  {'Domain':<25} {'TTA Error':>12}")
        print(f"  {'─'*25}─{'─'*12}")

        for domain_name, _ in domain_sequence:
            t_err = tta_errors.get(domain_name, 0)
            print(f"  {domain_name:<25} {t_err:>11.1f}%")

        tta_mean = np.mean(list(tta_errors.values()))
        print(f"  {'─'*25}─{'─'*12}")
        print(f"  {'MEAN':<25} {tta_mean:>11.1f}%")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
