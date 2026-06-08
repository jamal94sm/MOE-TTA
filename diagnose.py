"""
diagnose3.py — Detailed per-domain diagnostics with anti-collapse fixes.
Prints collapse indicators: prediction diversity, filter pass rate,
shared expert drift, and per-batch entropy distribution.

Usage:
  # Without fixes (reproduce collapse):
  python diagnose3.py --data_dir ./data/ImageNet-C --severity 5

  # With entropy floor only:
  python diagnose3.py --data_dir ./data/ImageNet-C --severity 5 --entropy_floor 0.05

  # With stochastic restore only:
  python diagnose3.py --data_dir ./data/ImageNet-C --severity 5 --stochastic_restore 0.01

  # With both fixes:
  python diagnose3.py --data_dir ./data/ImageNet-C --severity 5 --entropy_floor 0.05 --stochastic_restore 0.01
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

    # build model + FDD
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

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # ─── Save initial shared expert state ───
    # For drift measurement (always) and stochastic restore (if enabled)
    init_shared_state = {}
    init_shared_norms = {}
    for i, em in enumerate(model.expert_modules):
        for name, p in em.shared_moe.named_parameters():
            key = f"block{i}.shared.{name}"
            init_shared_state[key] = p.data.clone()
            init_shared_norms[key] = p.data.norm().item()

    print(f"\n{'='*80}")
    print(f"  DETAILED DIAGNOSTIC: {len(domain_sequence)} domains")
    print(f"  Entropy threshold κ = {cfg.entropy_threshold:.4f}")
    print(f"  Entropy floor       = {cfg.entropy_floor}")
    print(f"  Stochastic restore  = {cfg.stochastic_restore}")
    print(f"{'='*80}\n")

    for seg_idx, (domain_name, loader) in enumerate(domain_sequence):
        n_batches = len(loader)

        # ─── Per-domain accumulators ──
        all_errors = []
        all_entropies = []
        all_filter_rates = []
        all_pred_classes = []
        updates_applied = 0
        updates_skipped = 0
        floor_filtered = 0

        print(f"\n{'─'*80}")
        print(f"  [{seg_idx+1}/{len(domain_sequence)}] {domain_name} "
              f"({len(loader.dataset)} samples, {n_batches} batches)")
        print(f"{'─'*80}")

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
                params = model.get_trainable_params()
                optimizer = torch.optim.Adam(params, lr=cfg.lr,
                                             weight_decay=cfg.weight_decay)
                if batch_idx == 0:
                    print(f"  [FDD] New domain {fdd_domain_id} → expert {expert_id}")
            elif fdd_domain_id != current_fdd_domain:
                model.set_active_domain(fdd_domain_id)
                current_fdd_domain = fdd_domain_id
                params = model.get_trainable_params()
                optimizer = torch.optim.Adam(params, lr=cfg.lr,
                                             weight_decay=cfg.weight_decay)
                if batch_idx < 5:
                    print(f"  [FDD] Switched to domain {fdd_domain_id} at batch {batch_idx}")

            # ─── Forward ───
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B]
            preds = logits.argmax(dim=-1)

            # ─── Metrics ───
            err = 100.0 * (1 - (preds == labels).float().mean().item())
            all_errors.append(err)
            all_entropies.append(entropy.mean().item())

            unique_preds = len(set(preds.cpu().tolist()))
            all_pred_classes.append(unique_preds)

            # ─── Confidence filter with entropy floor ───
            mask = entropy < cfg.entropy_threshold
            if cfg.entropy_floor > 0:
                floor_mask = entropy > cfg.entropy_floor
                n_floor_rejected = int((mask & ~floor_mask).sum().item())
                floor_filtered += n_floor_rejected
                mask = mask & floor_mask

            pass_rate = mask.float().mean().item() * 100
            all_filter_rates.append(pass_rate)

            # ─── Update (only if confident-but-not-overconfident samples exist) ───
            if mask.sum() > 0 and optimizer is not None:
                loss = entropy[mask].mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates_applied += 1

                # ─── Stochastic restore of shared expert ───
                if cfg.stochastic_restore > 0:
                    with torch.no_grad():
                        for i, em in enumerate(model.expert_modules):
                            for name, p in em.shared_moe.named_parameters():
                                key = f"block{i}.shared.{name}"
                                if key in init_shared_state:
                                    rst_mask = (torch.rand_like(p) <
                                                cfg.stochastic_restore)
                                    p.data = torch.where(
                                        rst_mask,
                                        init_shared_state[key],
                                        p.data)
            else:
                updates_skipped += 1

            # ─── Print detailed info ───
            if batch_idx < 5 or batch_idx % 200 == 0 or batch_idx == n_batches - 1:
                pred_counts = Counter(preds.cpu().tolist())
                top_pred, top_count = pred_counts.most_common(1)[0]
                top_pct = 100 * top_count / B

                print(f"    batch {batch_idx:4d} | err={err:5.1f}% | "
                      f"H={entropy.mean().item():.3f} | "
                      f"filter={pass_rate:5.1f}% | "
                      f"unique_preds={unique_preds:3d}/1000 | "
                      f"top_pred=cls{top_pred}({top_pct:.0f}%)")

        # ─── Domain summary ───
        avg_err = np.mean(all_errors)
        avg_H = np.mean(all_entropies)
        avg_filter = np.mean(all_filter_rates)
        avg_diversity = np.mean(all_pred_classes)

        q = len(all_errors) // 4
        if q > 0:
            first_q_err = np.mean(all_errors[:q])
            last_q_err = np.mean(all_errors[-q:])
            trend = last_q_err - first_q_err
        else:
            first_q_err = last_q_err = avg_err
            trend = 0

        # shared expert drift
        drift_total = 0
        drift_count = 0
        for i, em in enumerate(model.expert_modules):
            for name, p in em.shared_moe.named_parameters():
                key = f"block{i}.shared.{name}"
                if key in init_shared_norms:
                    drift = abs(p.data.norm().item() - init_shared_norms[key])
                    drift_total += drift
                    drift_count += 1
        avg_drift = drift_total / max(drift_count, 1)

        print(f"\n  ┌── DOMAIN SUMMARY: {domain_name}")
        print(f"  │ Avg Error:        {avg_err:.1f}%")
        print(f"  │ Avg Entropy:      {avg_H:.3f}  "
              f"(threshold κ={cfg.entropy_threshold:.3f})")
        print(f"  │ Avg Filter Rate:  {avg_filter:.1f}% of samples pass band "
              f"[{cfg.entropy_floor}, {cfg.entropy_threshold:.3f})")
        print(f"  │ Avg Pred Diversity:{avg_diversity:.0f} unique classes / batch")
        print(f"  │ Error Trend:      first25%={first_q_err:.1f}% → "
              f"last25%={last_q_err:.1f}% (Δ={trend:+.1f}%)")
        print(f"  │ Updates:          {updates_applied} applied, "
              f"{updates_skipped} skipped")
        if cfg.entropy_floor > 0:
            print(f"  │ Floor-filtered:   {floor_filtered} samples rejected "
                  f"(H < {cfg.entropy_floor})")
        if cfg.stochastic_restore > 0:
            print(f"  │ Stochastic restore: p={cfg.stochastic_restore} per step")
        print(f"  │ Shared Expert Drift: {avg_drift:.4f} "
              f"(avg param norm change from init)")
        print(f"  │ FDD Domain:       {current_fdd_domain} "
              f"(total discovered: {fdd.num_domains})")

        # ─── Collapse warnings ───
        if avg_diversity < 10:
            print(f"  │ ⚠ COLLAPSE: model predicts <10 unique classes!")
        if avg_H < 0.1:
            print(f"  │ ⚠ COLLAPSE: entropy near zero (overconfident)")
        if avg_filter > 95 and avg_err > 80:
            print(f"  │ ⚠ COLLAPSE: filter passes everything but error is high")
        if trend > 10:
            print(f"  │ ⚠ DEGRADING: error increased {trend:+.1f}% within domain")
        print(f"  └{'─'*60}")

    print(f"\n{'='*80}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
