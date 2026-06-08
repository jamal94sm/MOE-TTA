"""
config.py — All hyperparameters from AAAI 2026 paper (Appendix F).
"""

import argparse, math

# ─── Corruption sequences ────────────────────────────────────────────
IMAGENET_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CIFAR100_C_CORRUPTIONS = IMAGENET_C_CORRUPTIONS          # same 15 types

CRS_DOMAINS = ["imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch"]

ACDC_DOMAINS = ["fog", "night", "rain", "snow"]

# ─── Class mappings for partial-class datasets ────────────────────────
# ImageNet-A and ImageNet-R use 200 of the 1000 ImageNet classes.
# These mappings are loaded at runtime from the dataset folders.

# ─── Defaults ────────────────────────────────────────────────────────
def get_cfg(args=None):
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", default="imagenet_c",
                   choices=["imagenet_c", "cifar100_c",
                            "imagenet_plus", "imagenet_plusplus", "acdc"])
    p.add_argument("--data_dir", default="./data/ImageNet-C")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--corruptions", nargs="*", default=None,
                   help="Corruption(s) to run. None=all 15. "
                        "Examples: --corruptions glass_blur | "
                        "--corruptions glass_blur defocus_blur fog")
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_rounds", type=int, default=3,
                   help="Rounds for CRS benchmark")

    # backbone
    p.add_argument("--backbone", default="vit_base_patch16_224")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_classes", type=int, default=1000)

    # expert architecture (Section 3.2)
    p.add_argument("--shared_rank", type=int, default=32,
                   help="LoRA rank for shared expert branch")
    p.add_argument("--domain_rank", type=int, default=16,
                   help="LoRA rank for each domain self-adaptive expert")
    p.add_argument("--num_experts_per_moe", type=int, default=2,
                   help="M: number of experts inside each MoE module")
    p.add_argument("--fusion_lambda", type=float, default=0.5,
                   help="λ: balance shared vs domain branch (Eq.5)")

    # FDD (Section 3.3)
    p.add_argument("--fdd_freq_radius", type=int, default=16,
                   help="l: frequency radius for low-freq crop")
    p.add_argument("--fdd_threshold", type=float, default=1.5,
                   help="τ: Mahalanobis threshold for new domain")
    p.add_argument("--fdd_shrinkage", type=float, default=0.1,
                   help="ε: covariance shrinkage")
    p.add_argument("--fdd_init_var", type=float, default=1.0,
                   help="σ²₀: initial diagonal covariance for new domain")
    p.add_argument("--fdd_diagonal", action="store_true", default=True,
                   help="Approximate covariance as diagonal (saves ~60%% GPU mem)")

    # optimisation (Section 3.4, Appendix F)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (auto-set per dataset if None)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--confidence_threshold", type=float, default=0.4,
                   help="κ coefficient: threshold = κ × ln(C)")

    # anti-collapse fixes
    p.add_argument("--entropy_floor", type=float, default=0.0,
                   help="Skip update when entropy < this value "
                        "(prevents reinforcing overconfident predictions). "
                        "0 = disabled. Recommended: 0.05")
    p.add_argument("--stochastic_restore", type=float, default=0.0,
                   help="Probability of resetting each shared expert param "
                        "to its initial value each step (CoTTA-style). "
                        "0 = disabled. Recommended: 0.01")

    # misc
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output")

    cfg = p.parse_args(args)

    # ── auto-set learning rate per dataset ──
    if cfg.lr is None:
        lr_map = {
            "imagenet_c": 1e-5,
            "cifar100_c": 1e-5,
            "imagenet_plus": 5e-4,
            "imagenet_plusplus": 5e-4,
            "acdc": 3e-4,
        }
        cfg.lr = lr_map[cfg.dataset]

    # ── auto-set num_classes ──
    if cfg.dataset == "cifar100_c":
        cfg.num_classes = 100
        cfg.img_size = 384          # paper resizes CIFAR to 384
        cfg.backbone = "vit_base_patch16_384"

    # ── entropy threshold κ × ln(C) ──
    cfg.entropy_threshold = cfg.confidence_threshold * math.log(cfg.num_classes)

    return cfg
