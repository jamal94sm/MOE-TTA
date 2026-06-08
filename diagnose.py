"""
diagnose2.py — Follow-up: isolate the cause of entropy collapse.
Tests 4 configurations over 200 batches each and plots the error trajectory.

Usage:
  python diagnose2.py --data_dir ./data/ImageNet-C --corruptions gaussian_noise
"""

import torch
import torch.nn.functional as F
import timm
import numpy as np
from tqdm import tqdm

from config import get_cfg
from datasets import get_domain_sequence
from model import build_model


def run_adaptation(cfg, loader, device, label, lr_override=None,
                   shared_only=False, domain_only=False,
                   max_batches=200, disable_adaptation=False):
    """
    Run adaptation and track error every batch.
    Returns list of per-batch error rates.
    """
    model = build_model(cfg)
    model.expand_domain()
    model.set_active_domain(0)

    # optionally disable one branch
    if shared_only:
        # zero out domain expert contribution: set fusion_lambda=1.0
        for em in model.expert_modules:
            em.fusion_lambda = 1.0  # Z + 1.0*shared + 0.0*domain
    elif domain_only:
        for em in model.expert_modules:
            em.fusion_lambda = 0.0  # Z + 0.0*shared + 1.0*domain

    params = model.get_trainable_params()
    lr = lr_override if lr_override is not None else cfg.lr

    if not disable_adaptation:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = None

    errors = []
    for i, (imgs, labels) in enumerate(loader):
        if i >= max_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        # forward
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # error
        preds = logits.argmax(dim=-1)
        err = 100.0 * (1 - (preds == labels).float().mean().item())
        errors.append(err)

        # adapt (unless disabled)
        mask = entropy < cfg.entropy_threshold
        if mask.sum() > 0:
            loss = entropy[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 50 == 0:
            avg_so_far = np.mean(errors)
            print(f"  [{label}] batch {i:4d} | err={err:.1f}% | "
                  f"running_avg={avg_so_far:.1f}%")

    final_avg = np.mean(errors)
    print(f"  [{label}] FINAL avg error over {len(errors)} batches: {final_avg:.1f}%")
    return errors


def main():
    cfg = get_cfg()
    device = cfg.device
    domain_sequence = get_domain_sequence(cfg)
    domain_name, loader = domain_sequence[0]
    n_batches = 1000

    print(f"Diagnosing collapse on: {domain_name}")
    print(f"Running {n_batches} batches per configuration\n")

    results = {}

    # ─── Config A: No adaptation (frozen baseline) ────────────────────
    print("=" * 60)
    print("A: No adaptation (Source baseline, frozen)")
    print("=" * 60)
    results["A: No adaptation"] = run_adaptation(
        cfg, loader, device, "no-adapt",
        disable_adaptation=True, max_batches=n_batches)

    # ─── Config B: Paper's lr=1e-5 (current setting) ─────────────────
    print("\n" + "=" * 60)
    print(f"B: Full adaptation, lr={cfg.lr}")
    print("=" * 60)
    results[f"B: lr={cfg.lr}"] = run_adaptation(
        cfg, loader, device, f"lr={cfg.lr}",
        max_batches=n_batches)

    # ─── Config C: Much smaller lr ────────────────────────────────────
    small_lr = cfg.lr * 0.1   # 1e-6
    print("\n" + "=" * 60)
    print(f"C: Full adaptation, lr={small_lr}")
    print("=" * 60)
    results[f"C: lr={small_lr}"] = run_adaptation(
        cfg, loader, device, f"lr={small_lr}",
        lr_override=small_lr, max_batches=n_batches)

    # ─── Config D: Shared expert only (λ=1, no domain expert) ─────────
    print("\n" + "=" * 60)
    print(f"D: Shared expert only (λ=1.0), lr={cfg.lr}")
    print("=" * 60)
    results["D: Shared only"] = run_adaptation(
        cfg, loader, device, "shared-only",
        shared_only=True, max_batches=n_batches)

    # ─── Config E: Domain expert only (λ=0, no shared expert) ─────────
    print("\n" + "=" * 60)
    print(f"E: Domain expert only (λ=0.0), lr={cfg.lr}")
    print("=" * 60)
    results["E: Domain only"] = run_adaptation(
        cfg, loader, device, "domain-only",
        domain_only=True, max_batches=n_batches)

    # ─── Config F: Larger lr (paper's ImageNet+ rate) ─────────────────
    large_lr = 5e-4
    print("\n" + "=" * 60)
    print(f"F: Full adaptation, lr={large_lr}")
    print("=" * 60)
    results[f"F: lr={large_lr}"] = run_adaptation(
        cfg, loader, device, f"lr={large_lr}",
        lr_override=large_lr, max_batches=n_batches)

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: Average error over all batches")
    print("=" * 60)
    for name, errors in results.items():
        avg = np.mean(errors)
        first_50 = np.mean(errors[:50])
        last_50 = np.mean(errors[-50:])
        trend = "↑ collapsing" if last_50 > first_50 + 3 else \
                "↓ improving" if last_50 < first_50 - 3 else "→ stable"
        print(f"  {name:<30} avg={avg:5.1f}%  "
              f"first50={first_50:5.1f}%  last50={last_50:5.1f}%  {trend}")

    # ─── Save plot ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for name, errors in results.items():
            # smooth with rolling average for readability
            window = 10
            smoothed = np.convolve(errors, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=name, alpha=0.8)

        ax.set_xlabel("Batch")
        ax.set_ylabel("Error (%)")
        ax.set_title(f"Adaptation dynamics on {domain_name} — {n_batches} batches")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        fig.tight_layout()
        fig.savefig("diagnose2_collapse.png", dpi=150)
        plt.close(fig)
        print(f"\n  Plot saved: diagnose2_collapse.png")
    except Exception as e:
        print(f"\n  Could not save plot: {e}")


if __name__ == "__main__":
    main()
