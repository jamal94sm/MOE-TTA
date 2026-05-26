# ==============================================================
#  visualize_fft_domains.py
#
#  Visualises the low-frequency FFT amplitude of CASIA-MS and XJTU
#  palmprint images to reveal domain separability across spectral bands.
#
#  Each image → Gaussian-masked FFT amplitude vector → 2D projection.
#  Points coloured by domain (spectrum or device/lighting condition).
#
#  Projections:
#    1. PCA  — fast linear projection (good for global structure)
#    2. t-SNE — non-linear, preserves local clusters
#    3. UMAP  — non-linear, preserves both local and global structure
#
#  Also produces:
#    4. Mean amplitude heatmaps per domain
#    5. Pairwise domain distance matrix (cosine distance on mean descriptors)
#
#  Usage:
#    python visualize_fft_domains.py --dataset casiams
#    python visualize_fft_domains.py --dataset xjtu
#    python visualize_fft_domains.py --dataset casiams --beta 0.05
#    python visualize_fft_domains.py --dataset casiams --max_per_domain 100
#
#  Requirements: pip install scikit-learn matplotlib umap-learn
# ==============================================================

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from PIL import Image
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


# ══════════════════════════════════════════════════════════════
#  DATASET PARSERS
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root):
    """spectrum → {identity: [path, ...]}"""
    data     = defaultdict(lambda: defaultdict(list))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(data_root)):
        if Path(fname).suffix.lower() not in img_exts:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            continue
        identity = f"{parts[0]}_{parts[1]}"
        spectrum = parts[2]
        data[spectrum][identity].append(os.path.join(data_root, fname))
    return data   # {spectrum: {identity: [paths]}}


CASIAMS_ROOT = "/home/pai-ng/Jamal/CASIA-MS-ROI"
XJTU_ROOT    = "/home/pai-ng/Jamal/XJTU-UP"

XJTU_VARIATIONS = [
    ("iPhone", "Flash"),
    ("iPhone", "Nature"),
    ("huawei", "Flash"),
    ("huawei", "Nature"),
]

def parse_xjtu(data_root):
    """
    Returns {domain_label: {identity: [path, ...]}}
    where domain_label = "iPhone/Flash" etc.
    """
    IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}
    data     = defaultdict(lambda: defaultdict(list))

    for device, condition in XJTU_VARIATIONS:
        label   = f"{device}/{condition}"
        var_dir = os.path.join(data_root, device, condition)
        if not os.path.isdir(var_dir):
            print(f"  [XJTU] WARNING: {var_dir} not found")
            continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir = os.path.join(var_dir, id_folder)
            if not os.path.isdir(id_dir):
                continue
            parts = id_folder.split("_")
            if len(parts) < 2 or parts[0].upper() not in ("L", "R"):
                continue
            for fname in sorted(os.listdir(id_dir)):
                if Path(fname).suffix.lower() not in IMG_EXTS:
                    continue
                data[label][id_folder].append(os.path.join(id_dir, fname))

    return data   # {domain_label: {identity: [paths]}}


# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTES
# ══════════════════════════════════════════════════════════════

CASIAMS_COLORS = {
    "460" : "#4477EE",
    "630" : "#EE4444",
    "700" : "#FF8800",
    "850" : "#9944CC",
    "940" : "#22AA44",
    "WHT" : "#888888",
}
CASIAMS_LABELS = {
    "460" : "460nm (blue visible)",
    "630" : "630nm (red visible)",
    "700" : "700nm (deep red)",
    "850" : "850nm (NIR)",
    "940" : "940nm (NIR)",
    "WHT" : "WHT (broadband white)",
}

XJTU_COLORS = {
    "iPhone/Flash"  : "#4477EE",
    "iPhone/Nature" : "#22AA44",
    "huawei/Flash"  : "#EE4444",
    "huawei/Nature" : "#FF8800",
}
XJTU_LABELS = {
    "iPhone/Flash"  : "iPhone — Flash",
    "iPhone/Nature" : "iPhone — Natural light",
    "huawei/Flash"  : "Huawei — Flash",
    "huawei/Nature" : "Huawei — Natural light",
}


def get_palette(dataset):
    if dataset == "casiams":
        return CASIAMS_COLORS, CASIAMS_LABELS
    return XJTU_COLORS, XJTU_LABELS


# ══════════════════════════════════════════════════════════════
#  FFT DESCRIPTOR
# ══════════════════════════════════════════════════════════════

try:
    from scipy.ndimage import gaussian_filter
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def hard_mask(H, W, beta):
    """Binary circular mask. Radius = beta × min(H,W)."""
    cy, cx = H // 2, W // 2
    ys     = np.arange(H) - cy
    xs     = np.arange(W) - cx
    xs, ys = np.meshgrid(xs, ys)
    radius = min(H, W) * beta
    return ((xs**2 + ys**2) <= radius**2).astype(np.float32)


def radial_profile(amp_2d, n_bins=64):
    """Mean log-amplitude over n_bins concentric rings.
    Rotation-invariant, identity-agnostic, n_bins-d."""
    H, W   = amp_2d.shape
    cy, cx = H // 2, W // 2
    Y, X   = np.ogrid[:H, :W]
    R      = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_max  = min(H, W) / 2.0
    bins   = np.zeros(n_bins, dtype=np.float32)
    edges  = np.linspace(0, r_max, n_bins + 1)
    for b in range(n_bins):
        ring = (R >= edges[b]) & (R < edges[b + 1])
        if ring.any():
            bins[b] = amp_2d[ring].mean()
    return bins


def extract_descriptor(path, img_side, beta, n_bins=64, mode="radial"):
    """
    Two descriptor modes (toggled via --desc_mode):

    "radial"  (recommended — best PCA separation in experiments):
      log(1 + |FFT(image)|) → hard circular mask → radial profile
      Captures spectral decay rate differences between domains.
      460nm vs NIR bands differ in rolloff → clearly separated in PCA.
      Descriptor: n_bins-d (default 64-d).

    "raw"  (baseline for comparison):
      |FFT(image)| → hard circular mask → flatten
      Raw low-frequency amplitude, no log transform.
      Descriptor: (2*r)²-d where r = beta*img_side (e.g. 16384-d at beta=0.5).

    Returns (raw_amp_for_heatmap, descriptor_vector).
    """
    img    = Image.open(path).convert("L").resize(
        (img_side, img_side), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    amp    = np.fft.fftshift(np.abs(np.fft.fft2(img_np)))   # raw for heatmap

    if mode == "raw":
        mask = hard_mask(img_side, img_side, beta)
        desc = (amp * mask).flatten()
    else:   # "radial"
        amp_log = np.log1p(amp)
        mask    = hard_mask(img_side, img_side, beta)
        masked  = amp_log * mask
        desc    = radial_profile(masked, n_bins=n_bins)

    return amp, desc



# ══════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════

def plot_scatter(coords, domain_labels_list, domains, title, ax,
                 colors, labels_map, alpha=0.5, size=12):
    for sp in domains:
        mask = np.array(domain_labels_list) == sp
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors.get(sp, "#333333"),
                       label=labels_map.get(sp, sp),
                       alpha=alpha, s=size, linewidths=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)


def plot_mean_heatmaps(mean_amps, domains, img_side, beta,
                       colors, labels_map, out_path):
    n     = len(domains)
    mask  = hard_mask(img_side, img_side, beta)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5*ncols, 4*nrows))
    axes = np.array(axes).flatten()

    # apply log transform for display too
    log_amps = {sp: np.log1p(mean_amps[sp]) * mask for sp in domains}
    vmax = max(a.max() for a in log_amps.values())

    for ax, sp in zip(axes, domains):
        im = ax.imshow(log_amps[sp], cmap="inferno",
                       vmin=0, vmax=vmax,
                       origin="upper")
        ax.set_title(labels_map.get(sp, sp), fontsize=10,
                     color=colors.get(sp, "#333"))
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Mean Low-Frequency FFT Amplitude per Domain\n"
        f"log(1+amp) · hard circular mask  (β={beta})",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_distance_matrix(mean_descs, domains, labels_map, out_path):
    """Pairwise cosine distance between domain mean descriptors."""
    n    = len(domains)
    vecs = np.stack([mean_descs[sp] for sp in domains])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    dist = 1.0 - (norm @ norm.T)

    fig, ax = plt.subplots(figsize=(max(5, n*1.2), max(4, n*1.0)))
    im      = ax.imshow(dist, cmap="RdYlGn_r", vmin=0, vmax=dist.max())
    short   = [labels_map.get(sp, sp).split("(")[0].split("—")[0].strip()
               for sp in domains]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist[i,j]:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if dist[i,j] > dist.max()*0.5 else "black")
    plt.colorbar(im, ax=ax, label="Cosine Distance")
    ax.set_title("Pairwise Domain Distance\n(radial FFT profile, log-amplitude)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



def classify_domains(descriptors, domain_labels_list, domains,
                     classifier="svm", desc_mode="radial",
                     out_path=None, seed=42):
    """
    Train a domain classifier and report 5-fold cross-validated accuracy.

    Classifiers:
      "svm" — RBF SVM. Best for small-to-medium datasets, nonlinear boundaries.
      "nn"  — MLP (2 hidden layers). Better for high-dimensional raw descriptors.

    Reports: per-class accuracy, confusion matrix figure.
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline

    X  = np.stack(descriptors).astype(np.float32)
    le = LabelEncoder()
    le.fit(domains)
    y  = le.transform(domain_labels_list)

    if classifier == "svm":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10.0, gamma="scale",
                           random_state=seed, decision_function_shape="ovr")),
        ])
    else:  # nn
        hidden = (256, 128) if X.shape[1] > 100 else (128, 64)
        model  = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(hidden_layer_sizes=hidden,
                                     max_iter=500, random_state=seed,
                                     early_stopping=True,
                                     n_iter_no_change=20)),
        ])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    acc    = accuracy_score(y, y_pred) * 100
    report = classification_report(y, y_pred,
                                   target_names=le.classes_, digits=3)
    cm     = confusion_matrix(y, y_pred)

    print("\n" + "="*60)
    print(f"  Domain Classifier — {classifier.upper()}  |  "
          f"descriptor: {desc_mode}  |  dim={X.shape[1]}")
    print(f"  5-fold CV accuracy: {acc:.2f}%")
    print("=" * 60)
    print(report)

    if out_path is not None:
        fig, ax = plt.subplots(figsize=(max(5, len(domains)*0.9),
                                        max(4, len(domains)*0.8)))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im      = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        labels_s = [l.split("(")[0].split("—")[0].strip() for l in le.classes_]
        ax.set_xticks(range(len(domains))); ax.set_yticks(range(len(domains)))
        ax.set_xticklabels(labels_s, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels_s, fontsize=9)
        for i in range(len(domains)):
            for j in range(len(domains)):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        fontsize=9,
                        color="white" if cm_norm[i,j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="Normalised count")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Domain Classifier — {classifier.upper()}\n"
                     f"Descriptor: {desc_mode} ({X.shape[1]}-d)  "
                     f"Acc={acc:.1f}%  5-fold CV",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    return acc, report


def plot_radial_profiles(mean_descs, domains, colors, labels_map, beta, out_path):
    """
    Plot mean radial spectral profile per domain as overlaid line curves.
    X-axis = frequency bin (low → high), Y-axis = mean log-amplitude.
    Separable domains → clearly separated curves.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for sp in domains:
        profile = mean_descs[sp]
        ax.plot(profile, color=colors.get(sp, "#333"),
                label=labels_map.get(sp, sp), linewidth=2)

    ax.set_xlabel("Frequency bin (low → high frequency)", fontsize=11)
    ax.set_ylabel("Mean log(1 + amplitude)", fontsize=11)
    ax.set_title(f"Radial Spectral Profile per Domain  (β={beta})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



    n    = len(domains)
    vecs = np.stack([mean_descs[sp] for sp in domains])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    dist = 1.0 - (norm @ norm.T)

    fig, ax = plt.subplots(figsize=(max(5, n*1.2), max(4, n*1.0)))
    im      = ax.imshow(dist, cmap="RdYlGn_r", vmin=0, vmax=dist.max())
    short   = [labels_map.get(sp, sp).split("(")[0].split("—")[0].strip()
               for sp in domains]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist[i,j]:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if dist[i,j] > dist.max()*0.5 else "black")

    plt.colorbar(im, ax=ax, label="Cosine Distance")
    ax.set_title("Pairwise Domain Distance\n(low-frequency FFT descriptors)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_projections(descriptors, domain_labels_list, domains, methods,
                     colors, labels_map, dataset_name, beta, n_total,
                     out_path, seed=42):
    X    = np.stack(descriptors)
    X    = normalize(X, norm="l2")

    MAX_TSNE = 2000
    if len(X) > MAX_TSNE and ("tsne" in methods or "umap" in methods):
        rng    = np.random.RandomState(seed)
        idx    = rng.choice(len(X), MAX_TSNE, replace=False)
        X_sub  = X[idx]
        dl_sub = [domain_labels_list[i] for i in idx]
        print(f"  Subsampled to {MAX_TSNE} for t-SNE/UMAP (from {len(X)})")
    else:
        X_sub, dl_sub = X, domain_labels_list

    results = {}

    if "pca" in methods:
        results["PCA"] = (PCA(n_components=2, random_state=seed)
                          .fit_transform(X), domain_labels_list)

    if "tsne" in methods:
        n_pca      = min(50, X_sub.shape[1], X_sub.shape[0] - 1)
        X_pca      = PCA(n_components=n_pca, random_state=seed).fit_transform(X_sub)
        perp       = min(30, X_sub.shape[0] // 4)
        import sklearn
        kw = dict(n_components=2, perplexity=perp, random_state=seed, verbose=0)
        sv = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        kw["max_iter" if sv >= (1, 5) else "n_iter"] = 1000
        results["t-SNE"] = (TSNE(**kw).fit_transform(X_pca), dl_sub)
        print(f"    t-SNE done (perplexity={perp})")

    if "umap" in methods:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed,
                                n_neighbors=15, min_dist=0.1)
            results["UMAP"] = (reducer.fit_transform(X_sub), dl_sub)
            print("    UMAP done")
        except ImportError:
            print("    UMAP not installed — skipping (pip install umap-learn)")

    n_plots = len(results)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6.5*n_plots, 6.5))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, (coords, lbls)) in zip(axes, results.items()):
        plot_scatter(coords, lbls, domains, name, ax,
                     colors, labels_map, alpha=0.55, size=14)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors.get(sp, "#333"),
               markersize=9, label=labels_map.get(sp, sp))
        for sp in domains
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(domains), 3), fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)

    fig.suptitle(
        f"Residual FFT + Whitening + Band Ratios — Domain Separability\n"
        f"{dataset_name.upper()}  |  β={beta}  |  {n_total} images",
        fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",        default="casiams",
                   choices=["casiams", "xjtu"],
                   help="dataset to visualise")
    p.add_argument("--data_root",      default=None,
                   help="path to dataset root (default: hardcoded CASIAMS_ROOT / XJTU_ROOT)")
    p.add_argument("--beta",           type=float, default=0.4,
                   help="hard mask radius as fraction of image size")
    p.add_argument("--n_bins",         type=int,   default=64,
                   help="number of radial profile bins (radial mode only)")
    p.add_argument("--desc_mode",      default="radial",
                   choices=["radial", "raw"],
                   help="radial=log+radial profile (recommended); raw=flat masked amplitude")
    p.add_argument("--classifier",     default="svm",
                   choices=["svm", "nn", "both"],
                   help="domain classifier: svm | nn | both")
    p.add_argument("--img_side",       type=int,   default=128)
    p.add_argument("--max_per_domain", type=int,   default=None,
                   help="max images per domain (None = all)")
    p.add_argument("--method",         nargs="+",
                   default=["pca", "tsne", "umap"],
                   choices=["pca", "tsne", "umap"])
    p.add_argument("--no_umap",        action="store_true")
    p.add_argument("--out_dir",        default="./plots")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.data_root is None:
        args.data_root = (CASIAMS_ROOT if args.dataset == "casiams"
                          else XJTU_ROOT)

    if args.no_umap and "umap" in args.method:
        args.method = [m for m in args.method if m != "umap"]

    os.makedirs(args.out_dir, exist_ok=True)
    colors, labels_map = get_palette(args.dataset)

    print(f"\n{'='*56}")
    print(f"  FFT Domain Visualisation — {args.dataset.upper()}")
    print(f"  data_root      : {args.data_root}")
    print(f"  beta           : {args.beta}")
    print(f"  n_bins         : {args.n_bins}  (radial mode)")
    print(f"  desc_mode      : {args.desc_mode}")
    print(f"  classifier     : {args.classifier}")
    print(f"  img_side       : {args.img_side}")
    print(f"  max_per_domain : {args.max_per_domain or 'all'}")
    print(f"  methods        : {args.method}")
    print(f"  out_dir        : {args.out_dir}")
    print(f"{'='*56}\n")

    # ── 1. Parse ───────────────────────────────────────────────
    print(f"Parsing {args.dataset.upper()} …")
    if args.dataset == "casiams":
        data = parse_casia_ms(args.data_root)
    else:
        data = parse_xjtu(args.data_root)

    domains = (sorted(data.keys()) if args.dataset == "casiams"
               else [f"{d}/{c}" for d, c in XJTU_VARIATIONS
                     if f"{d}/{c}" in data])
    for sp in domains:
        n_ids = len(data[sp])
        n_img = sum(len(v) for v in data[sp].values())
        print(f"  {sp:>18}  IDs={n_ids}  images={n_img}")

    # ── 2. Extract descriptors ─────────────────────────────────
    print(f"\nExtracting FFT descriptors (β={args.beta}) …")
    descriptors       = []
    domain_labels_arr = []
    mean_amps         = {}
    mean_descs        = {}

    for sp in domains:
        paths = [p for paths in data[sp].values() for p in paths]
        if args.max_per_domain is not None:
            rng   = np.random.RandomState(args.seed)
            paths = list(rng.choice(paths,
                                     min(args.max_per_domain, len(paths)),
                                     replace=False))
        sp_descs, sp_amps = [], []
        for path in paths:
            try:
                amp, desc = extract_descriptor(
                path, args.img_side, args.beta,
                n_bins=args.n_bins, mode=args.desc_mode)
                descriptors.append(desc)
                domain_labels_arr.append(sp)
                sp_descs.append(desc)
                sp_amps.append(amp)
            except Exception as e:
                print(f"    [WARN] {path}: {e}")

        if sp_descs:
            mean_descs[sp] = np.stack(sp_descs).mean(axis=0)
            mean_amps[sp]  = np.stack(sp_amps).mean(axis=0)
            print(f"  {sp:>18}  {len(sp_descs)} descriptors")

    print(f"\n  Total: {len(descriptors)} descriptors  "
          f"(dim={descriptors[0].shape[0]})")

    tag = f"{args.dataset}_beta{args.beta}"

    # ── 3. Scatter projections ─────────────────────────────────
    print("\nComputing projections …")
    plot_projections(
        descriptors, domain_labels_arr, domains, args.method,
        colors, labels_map, args.dataset, args.beta, len(descriptors),
        out_path=os.path.join(args.out_dir, f"fft_scatter_{tag}.png"),
        seed=args.seed)

    # ── 4. Mean amplitude heatmaps ────────────────────────────
    print("\nPlotting mean amplitude heatmaps …")
    plot_mean_heatmaps(
        mean_amps, domains, args.img_side, args.beta,
        colors, labels_map,
        out_path=os.path.join(args.out_dir, f"fft_heatmaps_{tag}.png"))

    # ── 5. Pairwise distance matrix ────────────────────────────
    print("Plotting pairwise distance matrix …")
    plot_distance_matrix(
        mean_descs, domains, labels_map,
        out_path=os.path.join(args.out_dir, f"fft_distances_{tag}.png"))

    # ── 6. Radial profile curves ───────────────────────────────
    if args.desc_mode == "radial":
        print("Plotting radial spectral profiles …")
        plot_radial_profiles(
            mean_descs, domains, colors, labels_map, args.beta,
            out_path=os.path.join(args.out_dir, f"fft_radial_{tag}.png"))

    # ── 7. Domain classifier ───────────────────────────────────
    clf_names = (["svm", "nn"] if args.classifier == "both"
                 else [args.classifier])
    print(f"\nTraining domain classifier(s): {clf_names} …")
    results = {}
    for clf in clf_names:
        cm_path = os.path.join(
            args.out_dir,
            f"clf_confusion_{clf}_{args.desc_mode}_{tag}.png")
        acc, _ = classify_domains(
            descriptors, domain_labels_arr, domains,
            classifier  = clf,
            desc_mode   = args.desc_mode,
            out_path    = cm_path,
            seed        = args.seed)
        results[clf] = acc

    print(f"\n  ── Classifier Summary ──────────────────────────")
    print(f"  Dataset    : {args.dataset.upper()}")
    print(f"  Descriptor : {args.desc_mode}  ({len(descriptors[0])}-d)")
    for clf, acc in results.items():
        print(f"  {clf.upper():<6} accuracy : {acc:.2f}%")
    print(f"  ──────────────────────────────────────────────")

    # ── 6. Console summary ─────────────────────────────────────
    print("\nPairwise cosine distances:")
    vecs = np.stack([mean_descs[sp] for sp in domains])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    dist = 1.0 - (norm @ norm.T)
    w    = max(len(sp) for sp in domains) + 2
    print(" " * w + "".join(f"{sp:>{w}}" for sp in domains))
    for i, sp_i in enumerate(domains):
        print(f"{sp_i:>{w}}" + "".join(f"{dist[i,j]:>{w}.4f}"
                                        for j in range(len(domains))))

    print(f"\nDone. Figures saved to: {args.out_dir}/")


import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from collections import defaultdict
from PIL import Image

# sklearn always available
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


# ── CASIA-MS dataset parser (mirrors datasets.py) ─────────────
