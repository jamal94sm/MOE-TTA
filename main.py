"""
main.py — Complete CTTA adaptation loop for the AAAI 2026 paper.

Algorithm per batch:
  1. FDD detects domain (new or existing) from low-frequency descriptors
  2. If new domain: expand expert pool, set as active
  3. If known domain: set as active, freeze others
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
from tqdm import tqdm

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


# ─── Cosine LR helper (replaces CosineAnnealingLR) ───────────────────

def cosine_lr(base_lr, step, total_steps):
    """Compute LR at a given step using cosine annealing."""
    if total_steps <= 0:
        return base_lr
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))


# ─── Entropy loss with confidence filtering (Eq. 18) ──────────────────

def filtered_entropy_loss(logits, threshold, entropy_floor=0.0):
    """
    L_TTA = -𝟙{floor < H(ŷ) < κ} Σ ŷ_c log ŷ_c

    logits:        [B, C] raw logits
    threshold:     κ = 0.4 × ln(C)  (ceiling: skip uncertain samples)
    entropy_floor: skip overconfident samples (prevents collapse reinforcement)
    Returns: scalar loss (mean over reliable samples), or 0 if none pass
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

    # mean entropy over reliable samples
    loss = (entropy * mask).sum() / mask.sum()
    return loss


# ─── Build optimizer for current trainable params ─────────────────────

def make_optimizer(model, cfg, current_lr):
    """Create Adam optimizer for currently trainable params at given LR."""
    params = model.get_trainable_params()
    if not params:
        return None
    return Adam(params, lr=current_lr,
                betas=(0.9, 0.999),
                weight_decay=cfg.weight_decay)


# ─── Main adaptation loop ────────────────────────────────────────────

def adapt(cfg):
    """Complete CTTA adaptation following the paper's algorithm."""

    print(f"\n{'='*70}")
    print(f"  AAAI 2026: Shared & Domain Self-Adaptive Experts with FDD")
    print(f"  Dataset: {cfg.dataset} | LR: {cfg.lr} | BS: {cfg.batch_size}")
    print(f"{'='*70}\n")

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
          f"stochastic_restore={cfg.stochastic_restore}")

    # ── cosine schedule over entire stream ──
    print(f"[Scheduler] Cosine annealing over {total_batches} total batches "
          f"(LR: {cfg.lr} → 0)")

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
    global_step = 0       # global batch counter across all domains

    # ── un-normalized transform for FDD (need raw pixel images) ──
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for seg_idx, (domain_name, loader) in enumerate(domain_sequence):
        seg_correct = 0
        seg_total = 0
        seg_loss_sum = 0.0
        t0 = time.time()

        print(f"[{seg_idx+1}/{len(domain_sequence)}] "
              f"Domain: {domain_name} ({len(loader.dataset)} samples)")

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
                print(f"  [FDD] New domain {fdd_domain_id} detected "
                      f"→ expert {expert_id} created")

                # new optimizer for the new set of trainable params
                current_lr = cosine_lr(cfg.lr, global_step, total_batches)
                optimizer = make_optimizer(model, cfg, current_lr)

            elif fdd_domain_id != current_fdd_domain:
                model.set_active_domain(fdd_domain_id)
                current_fdd_domain = fdd_domain_id

                if fdd_domain_id in domain_map:
                    domain_map[fdd_domain_id].append(domain_name)
                else:
                    domain_map[fdd_domain_id] = [domain_name]

                print(f"  [FDD] Matched domain {fdd_domain_id} "
                      f"→ expert {fdd_domain_id} activated")

                # new optimizer for changed trainable params
                current_lr = cosine_lr(cfg.lr, global_step, total_batches)
                optimizer = make_optimizer(model, cfg, current_lr)

            # ─── Update LR based on global step (cosine over full stream) ───
            if optimizer is not None:
                current_lr = cosine_lr(cfg.lr, global_step, total_batches)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # ─── Step 4: Forward pass ───
            logits = model(images)

            # ─── Step 5: Compute filtered entropy loss (Eq. 18) ───
            loss = filtered_entropy_loss(logits, cfg.entropy_threshold,
                                         cfg.entropy_floor)

            # ─── Step 6: Backward + update ───
            if optimizer is not None and loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ─── Stochastic restore of shared expert (CoTTA-style) ───
                if cfg.stochastic_restore > 0:
                    with torch.no_grad():
                        for i, em in enumerate(model.expert_modules):
                            for name, p in em.shared_moe.named_parameters():
                                key = f"{i}.{name}"
                                if key in shared_init_state:
                                    mask = (torch.rand_like(p) <
                                            cfg.stochastic_restore)
                                    p.data = torch.where(
                                        mask, shared_init_state[key], p.data)

            # ─── Step 7: Record predictions ───
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).sum().item()
                seg_correct += correct
                seg_total += B
                seg_loss_sum += loss.item() * B
                total_correct += correct
                total_samples += B

            global_step += 1

        # ── segment summary ──
        seg_acc = seg_correct / max(seg_total, 1) * 100
        seg_err = 100.0 - seg_acc
        seg_loss_avg = seg_loss_sum / max(seg_total, 1)
        elapsed = time.time() - t0

        results[domain_name] = {
            "error": seg_err,
            "accuracy": seg_acc,
            "loss": seg_loss_avg,
            "samples": seg_total,
            "fdd_domain": current_fdd_domain,
            "time": elapsed,
        }

        lr_now = cosine_lr(cfg.lr, global_step, total_batches)
        print(f"  Error: {seg_err:.1f}% | Acc: {seg_acc:.1f}% | "
              f"Loss: {seg_loss_avg:.4f} | "
              f"FDD domains: {fdd.num_domains} | "
              f"LR: {lr_now:.2e} | "
              f"Step: {global_step}/{total_batches} | "
              f"Time: {elapsed:.1f}s")

    # ─── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL RESULTS")
    print(f"{'='*70}")

    mean_error = np.mean([r["error"] for r in results.values()])
    mean_acc = np.mean([r["accuracy"] for r in results.values()])
    print(f"  Mean Error: {mean_error:.2f}%")
    print(f"  Mean Accuracy: {mean_acc:.2f}%")
    print(f"  Total FDD domains discovered: {fdd.num_domains}")

    # per-domain breakdown
    print(f"\n  Per-domain errors:")
    for name, r in results.items():
        print(f"    {name:30s} → {r['error']:.1f}%  "
              f"(FDD domain {r['fdd_domain']})")

    # CRS-specific: compute Repeat Forget (RF) metric
    if cfg.dataset in ["imagenet_plus", "imagenet_plusplus"]:
        _compute_rf(results, cfg)

    # save results
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir,
                            f"{cfg.dataset}_seed{cfg.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
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
