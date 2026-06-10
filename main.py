"""
main.py — Complete CTTA adaptation loop for the AAAI 2026 paper.

Algorithm per batch:
  1. FDD detects domain (new or existing) from low-frequency descriptors
  2. If new domain: expand expert pool, add params to optimizer
  3. If known domain: set as active (optimizer already has its params)
  4. Forward pass through backbone + dual-branch experts
  5. Compute filtered entropy loss (Eq. 18)
  6. Update trainable parameters (shared + active domain expert)
  7. Output predictions
"""

import os
import sys
import math
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from collections import Counter

from config import get_cfg
from model import build_model
from fdd import FrequencyDomainDiscriminator
from datasets import get_domain_sequence


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─── Entropy loss with confidence filtering (Eq. 18) + diversity ──────

def filtered_entropy_loss(logits, threshold, entropy_floor=0.0, div_lambda=0.0):
    """
    L = L_ent + λ_div × L_div

    L_ent: filtered per-sample entropy minimization (Eq. 18)
    L_div: batch diversity — negative entropy of mean prediction (IM loss)

    logits:        [B, C] raw logits
    threshold:     κ = 0.4 × ln(C)  (ceiling: skip uncertain samples)
    entropy_floor: skip overconfident samples (prevents collapse reinforcement)
    div_lambda:    weight for diversity term (0 = disabled)
    Returns: scalar loss, or 0 if no samples pass the filter
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # per-sample entropy
    entropy = -(probs * log_probs).sum(dim=-1)           # [B]

    # filter: keep samples in the safe band [floor, κ)
    mask = (entropy < threshold).float()                 # [B]
    if entropy_floor > 0:
        mask = mask * (entropy > entropy_floor).float()

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    # per-sample entropy minimization (filtered)
    ent_loss = (entropy * mask).sum() / mask.sum()

    # batch diversity: maximize entropy of mean prediction (all samples)
    if div_lambda > 0:
        batch_mean_prob = probs.mean(dim=0)              # [C]
        log_C = math.log(probs.shape[-1])
        div_loss = (batch_mean_prob * torch.log(batch_mean_prob + 1e-8)).sum() + log_C
        return ent_loss + div_lambda * div_loss

    return ent_loss


# ─── Main adaptation loop ────────────────────────────────────────────

def adapt(cfg):
    """Complete CTTA adaptation following the paper's algorithm."""

    print(f"\n{'='*90}")
    print(f"  AAAI 2026: Shared & Domain Self-Adaptive Experts with FDD")
    print(f"  Dataset: {cfg.dataset} | LR: {cfg.lr} | BS: {cfg.batch_size}")
    print(f"{'='*90}\n")

    # ── build model ──
    model = build_model(cfg)
    print(f"[Model] Backbone: {cfg.backbone}")
    print(f"[Model] Shared rank: {cfg.shared_rank}, "
          f"Domain rank: {cfg.domain_rank}, "
          f"Experts/MoE: {cfg.num_experts_per_moe}")

    # ── build FDD ──
    fdd = FrequencyDomainDiscriminator(
        freq_radius=cfg.fdd_freq_radius,
        threshold=cfg.fdd_threshold,
        shrinkage=cfg.fdd_shrinkage,
        init_var=cfg.fdd_init_var,
        diagonal=cfg.fdd_diagonal,
        device=cfg.device,
    )
    print(f"[FDD] freq_radius={cfg.fdd_freq_radius}, "
          f"threshold={cfg.fdd_threshold}, "
          f"diagonal={cfg.fdd_diagonal}")

    # ── build domain sequence ──
    domain_sequence = get_domain_sequence(cfg)
    total_batches = sum(len(loader) for _, loader in domain_sequence)
    print(f"[Data] {len(domain_sequence)} domain segments, "
          f"{total_batches} total batches")

    # ── anti-collapse settings ──
    print(f"[Anti-collapse] entropy_floor={cfg.entropy_floor}, "
          f"stochastic_restore={cfg.stochastic_restore}, "
          f"div_lambda={cfg.div_lambda}")
    print(f"[Optimizer] Constant LR={cfg.lr}, weight_decay={cfg.weight_decay}")

    # ── evaluate frozen backbone (source baseline) ──
    eval_bb = getattr(cfg, 'eval_backbone', False)
    baseline_errors = {}
    if eval_bb:
        print(f"\n[Baseline] Evaluating frozen backbone on each domain...")
        model.eval()
        with torch.no_grad():
            for domain_name, loader in domain_sequence:
                correct = 0
                total = 0
                for images, labels in loader:
                    images, labels = images.to(cfg.device), labels.to(cfg.device)
                    logits = model.backbone(images)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
                err = 100.0 * (1 - correct / total)
                baseline_errors[domain_name] = err
                print(f"  {domain_name:25s} → {err:.1f}%")
        backbone_mean = np.mean(list(baseline_errors.values()))
        print(f"[Baseline] Mean backbone error: {backbone_mean:.1f}%\n")

    # ── save initial shared expert state (for stochastic restore) ──
    shared_init_state = {}
    if cfg.stochastic_restore > 0:
        for i, em in enumerate(model.expert_modules):
            for name, p in em.shared_moe.named_parameters():
                shared_init_state[f"{i}.{name}"] = p.data.clone()

    # ── tracking ──
    optimizer = None
    results = {}
    domain_map = {}       # fdd_domain_id → list of domain_names
    total_correct = 0
    total_samples = 0
    current_fdd_domain = -1
    global_step = 0

    # ── un-normalized transform for FDD (need raw pixel images) ──
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # ── signal tracking header ──
    hdr = (f"  {'batch':>5} │ {'err%':>5} │ {'H_mean':>6} │ {'H_p10':>5} │ "
           f"{'H_p50':>5} │ {'H_p90':>5} │ {'filt%':>5} │ {'uniq':>4} │ "
           f"{'top_cls':>7} │ {'top%':>4} │ {'H_correct':>9} │ {'H_wrong':>7} │ "
           f"{'shrd_norm':>9} │ {'dom_norm':>8}")

    for seg_idx, (domain_name, loader) in enumerate(domain_sequence):
        seg_correct = 0
        seg_total = 0
        seg_loss_sum = 0.0
        n_batches = len(loader)
        t0 = time.time()

        # per-domain trackers
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
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            B = images.shape[0]

            # ─── Step 1: FDD domain detection ───
            mean = imagenet_mean.to(cfg.device)
            std = imagenet_std.to(cfg.device)
            raw_images = images * std + mean

            fdd_domain_id, is_new = fdd.detect_domain(raw_images)

            # ─── Step 2-3: Expand or activate domain expert ───
            if is_new:
                expert_id = model.expand_domain()
                model.set_active_domain(expert_id)
                current_fdd_domain = fdd_domain_id
                domain_map[fdd_domain_id] = [domain_name]
                print(f"  >>> FDD: New domain {fdd_domain_id} → expert {expert_id}")

                if optimizer is None:
                    optimizer = Adam(model.get_trainable_params(),
                                    lr=cfg.lr, betas=(0.9, 0.999),
                                    weight_decay=cfg.weight_decay)
                else:
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
                model.set_active_domain(fdd_domain_id)
                current_fdd_domain = fdd_domain_id

                if fdd_domain_id in domain_map:
                    domain_map[fdd_domain_id].append(domain_name)
                else:
                    domain_map[fdd_domain_id] = [domain_name]

                print(f"  >>> FDD: Matched domain {fdd_domain_id} "
                      f"→ expert {fdd_domain_id} activated")

            # ─── Step 4: Forward pass ───
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            preds = logits.argmax(dim=-1)

            # ─── Metrics ───
            correct = (preds == labels).sum().item()
            err = 100.0 * (1 - correct / B)
            all_errors.append(err)
            seg_correct += correct
            seg_total += B

            # entropy percentiles
            H_sorted = entropy.sort().values
            H_p10 = H_sorted[int(0.1 * B)].item()
            H_p50 = H_sorted[int(0.5 * B)].item()
            H_p90 = H_sorted[int(0.9 * B)].item()

            # entropy of correct vs wrong
            correct_mask = (preds == labels)
            H_correct = entropy[correct_mask].mean().item() if correct_mask.any() else -1
            H_wrong = entropy[~correct_mask].mean().item() if (~correct_mask).any() else -1

            # prediction diversity
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

            # ─── Step 5: Filter + entropy loss ───
            mask = entropy < cfg.entropy_threshold
            if cfg.entropy_floor > 0:
                mask = mask & (entropy > cfg.entropy_floor)
            pass_rate = mask.float().mean().item() * 100

            loss = filtered_entropy_loss(logits, cfg.entropy_threshold,
                                         cfg.entropy_floor, cfg.div_lambda)
            seg_loss_sum += loss.item() * B

            # ─── Step 6: Backward + update ───
            if optimizer is not None and loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates_applied += 1

                # stochastic restore of shared expert (CoTTA-style)
                if cfg.stochastic_restore > 0:
                    with torch.no_grad():
                        for i, em_r in enumerate(model.expert_modules):
                            for name, p in em_r.shared_moe.named_parameters():
                                key = f"{i}.{name}"
                                if key in shared_init_state:
                                    rst_mask = (torch.rand_like(p) <
                                                cfg.stochastic_restore)
                                    p.data = torch.where(
                                        rst_mask, shared_init_state[key], p.data)
            else:
                updates_skipped += 1

            # ─── Print signals every 50 batches, first 5, and last batch ───
            if batch_idx < 5 or batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                h_c_str = f"{H_correct:.3f}" if H_correct >= 0 else "  N/A"
                h_w_str = f"{H_wrong:.3f}" if H_wrong >= 0 else "  N/A"
                print(f"  {batch_idx:5d} │ {err:5.1f} │ {entropy.mean().item():6.3f} │ "
                      f"{H_p10:5.3f} │ {H_p50:5.3f} │ {H_p90:5.3f} │ "
                      f"{pass_rate:5.1f} │ {unique_preds:4d} │ "
                      f"cls{top_pred:>4d} │ {top_pct:4.0f} │ {h_c_str:>9s} │ "
                      f"{h_w_str:>7s} │ {shared_norm:9.4f} │ {domain_norm:8.4f}")

            total_correct += correct
            total_samples += B
            global_step += 1

        # ── domain summary ──
        seg_acc = seg_correct / max(seg_total, 1) * 100
        seg_err = 100.0 - seg_acc
        seg_loss_avg = seg_loss_sum / max(seg_total, 1)
        elapsed = time.time() - t0
        avg_err = np.mean(all_errors)

        results[domain_name] = {
            "error": seg_err,
            "accuracy": seg_acc,
            "loss": seg_loss_avg,
            "samples": seg_total,
            "fdd_domain": current_fdd_domain,
            "time": elapsed,
        }

        # error trend
        q = len(all_errors) // 4
        first_q = np.mean(all_errors[:q]) if q > 0 else avg_err
        last_q = np.mean(all_errors[-q:]) if q > 0 else avg_err
        trend = last_q - first_q

        # takeover detection
        takeover_batch = None
        for b_idx, (cls, pct) in enumerate(top_class_history):
            if pct > 50:
                takeover_batch = b_idx
                break

        # print summary
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
            print(f"  │ Avg Error:       {avg_err:.1f}%  "
                  f"(first25%={first_q:.1f}% → last25%={last_q:.1f}%, Δ={trend:+.1f}%)")
        print(f"  │ Loss: {seg_loss_avg:.4f} | "
              f"FDD domains: {fdd.num_domains} | "
              f"Time: {elapsed:.1f}s")
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

    # ─── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    if baseline_errors:
        print("  FINAL RESULTS: Backbone vs TTA")
    else:
        print("  FINAL RESULTS: TTA Adaptation Summary")
    print(f"{'='*90}")

    mean_error = np.mean([r["error"] for r in results.values()])

    if baseline_errors:
        print(f"\n  {'Domain':<25} {'Backbone':>10} {'TTA':>10} {'Improv.':>10} {'FDD':>5}")
        print(f"  {'─'*25}─{'─'*10}─{'─'*10}─{'─'*10}─{'─'*5}")

        for name, r in results.items():
            b_err = baseline_errors.get(name, 0)
            t_err = r["error"]
            imp = b_err - t_err
            arrow = "↓" if imp > 0 else "↑"
            marker = " ⚠" if imp < -1 else ""
            print(f"  {name:<25} {b_err:>9.1f}% {t_err:>9.1f}% "
                  f"{arrow} {abs(imp):>7.1f}%{marker:>3s} {r['fdd_domain']:>5d}")

        backbone_mean = np.mean(list(baseline_errors.values()))
        mean_imp = backbone_mean - mean_error
        print(f"  {'─'*25}─{'─'*10}─{'─'*10}─{'─'*10}─{'─'*5}")
        print(f"  {'MEAN':<25} {backbone_mean:>9.1f}% {mean_error:>9.1f}% "
              f"{'↓' if mean_imp > 0 else '↑'} {abs(mean_imp):>7.1f}%")
    else:
        print(f"\n  {'Domain':<25} {'TTA Error':>12} {'FDD Domain':>12}")
        print(f"  {'─'*25}─{'─'*12}─{'─'*12}")

        for name, r in results.items():
            print(f"  {name:<25} {r['error']:>11.1f}% {r['fdd_domain']:>12d}")

        print(f"  {'─'*25}─{'─'*12}─{'─'*12}")
        print(f"  {'MEAN':<25} {mean_error:>11.1f}%")

    print(f"\n  Total FDD domains discovered: {fdd.num_domains}")

    # CRS-specific: compute Repeat Forget (RF) metric
    if cfg.dataset in ["imagenet_plus", "imagenet_plusplus"]:
        _compute_rf(results, cfg)

    # ─── Save results ───
    save_data = {
        "tta": {name: r for name, r in results.items()},
        "mean_tta_error": mean_error,
        "total_fdd_domains": fdd.num_domains,
    }
    if baseline_errors:
        save_data["baseline"] = baseline_errors
        save_data["mean_backbone_error"] = backbone_mean
        save_data["mean_improvement"] = mean_imp

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir,
                            f"{cfg.dataset}_seed{cfg.seed}.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # domain mapping summary
    print(f"\n  FDD domain mapping:")
    for fdd_id, names in domain_map.items():
        unique_names = list(set(n.rsplit("_R", 1)[0] for n in names))
        print(f"    FDD {fdd_id} → {unique_names}")

    return results


def _compute_rf(results, cfg):
    """
    Compute Repeat Forget (RF) metric for CRS benchmarks.
    RF = Error_final - Error_first (lower is better).
    """
    from collections import defaultdict

    domain_rounds = defaultdict(list)
    for name, r in results.items():
        base = name.rsplit("_R", 1)[0]
        domain_rounds[base].append(r["error"])

    print(f"\n  Repeat Forget (RF) analysis:")
    rf_values = []
    for base, errors in domain_rounds.items():
        if len(errors) >= 2:
            rf = errors[-1] - errors[0]
            rf_values.append(rf)
            trend = "↑ forgot" if rf > 0 else "↓ improved"
            print(f"    {base:20s}: R1={errors[0]:.1f}% → "
                  f"R{len(errors)}={errors[-1]:.1f}%  "
                  f"RF={rf:+.1f} ({trend})")

    if rf_values:
        mean_rf = np.mean(rf_values)
        print(f"    {'Mean RF':20s}: {mean_rf:+.1f}")


# ─── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_cfg()
    set_seed(cfg.seed)
    adapt(cfg)
