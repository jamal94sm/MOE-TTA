# ==============================================================
#  visualize_fft_domains.py
#
#  Visualises FFT-based domain separability for CASIA-MS and XJTU
#  palmprint datasets. Answers the question: "do images from different
#  spectral domains / devices look different in the frequency domain?"
#
#  Three descriptor modes:
#    "radial"      — log(1+|FFT|) → radial ring means (best separation)
#    "raw"         — raw |FFT| flattened (baseline, very high-dim)
#    "sensorprint" — residual FFT + whitening + band ratios (forensic)
#
#  Five outputs per run (saved to OUT_DIR):
#    1. fft_scatter_*    — PCA / t-SNE / UMAP scatter (each point = 1 image)
#    2. fft_heatmaps_*   — mean FFT amplitude per domain (shows spectral shape)
#    3. fft_distances_*  — pairwise cosine distance matrix between domains
#    4. fft_radial_*     — mean radial spectral profile curves per domain
#    5. clf_confusion_*  — confusion matrix from domain classifier (SVM/NN)
#
#  All parameters are set as variables at the top of __main__ — no CLI needed.
#
#  Requirements: pip install scikit-learn matplotlib umap-learn scipy
# ==============================================================


import os
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


def _radial_distance_map(H, W):
    cy, cx = H // 2, W // 2
    Y, X   = np.ogrid[:H, :W]
    return np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)


def _band_energy_ratios(amp_2d, R, r_max):
    """6 frequency band energy ratios — normalisation-invariant."""
    def E(r0, r1):
        m = (R >= r0 * r_max) & (R < r1 * r_max)
        return float(amp_2d[m].mean()) if m.any() else 1e-8
    eps  = 1e-8
    E_ul = E(0.00, 0.05); E_l = E(0.05, 0.15)
    E_m  = E(0.15, 0.35); E_h = E(0.35, 0.65); E_uh = E(0.65, 1.00)
    return np.array([E_h/(E_l+eps),  E_uh/(E_l+eps),  E_h/(E_ul+eps),
                     E_m/(E_l+eps),  E_uh/(E_m+eps),  E_h/(E_m+eps)],
                    dtype=np.float32)


def extract_descriptor(path, img_side, beta, n_bins=64, mode="radial",
                       alpha=1.0):
    """
    Three descriptor modes (--desc_mode):

    "radial"      — best PCA separation (recommended):
      log(1+|FFT(image)|) → hard circular mask → radial profile (n_bins-d).
      Captures spectral decay rate; 460nm vs NIR clearly separated in PCA.

    "raw"         — baseline:
      |FFT(image)| → hard circular mask → flatten.
      Raw amplitude, no log transform, very high-dimensional.

    "sensorprint" — forensic-style sensor fingerprint (residual+whitening):
      Step 1 — residual = image − GaussianBlur(σ=2)
               Removes palm identity/content (low-freq spatial structure).
               Leaves device/sensor acquisition artifacts.
      Step 2 — FFT → log(1+|FFT(residual)|)
      Step 3 — whiten: amp_white = amp_log / r^α
               Removes universal 1/f² natural-image prior.
               Domain deviations from this prior become the signal.
      Step 4 — hard_mask × radial_profile (n_bins-d)
      Step 5 — band_energy_ratios (6-d)
      concat → (n_bins+6)-d descriptor.

    Returns (raw_amp_for_heatmap, descriptor_vector).
    """
    img    = Image.open(path).convert("L").resize(
        (img_side, img_side), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    amp    = np.fft.fftshift(np.abs(np.fft.fft2(img_np)))   # raw for heatmap

    if mode == "raw":
        mask = hard_mask(img_side, img_side, beta)
        desc = (amp * mask).flatten()

    elif mode == "radial":
        amp_log = np.log1p(amp)
        mask    = hard_mask(img_side, img_side, beta)
        masked  = amp_log * mask
        desc    = radial_profile(masked, n_bins=n_bins)

    else:  # "sensorprint"
        # Step 1 — residual spectrum
        if _SCIPY_OK:
            blurred  = gaussian_filter(img_np, sigma=2.0)
            residual = img_np - blurred
        else:
            residual = img_np
        amp_res = np.fft.fftshift(np.abs(np.fft.fft2(residual)))
        amp_log = np.log1p(amp_res)

        # Step 2 — power spectrum whitening (remove 1/f^alpha prior)
        H, W    = amp_log.shape
        R       = _radial_distance_map(H, W)
        eps_w   = 1e-6
        divisor = np.maximum(R ** alpha, eps_w)
        amp_white           = amp_log / divisor
        amp_white[R < 1]    = 0.0         # suppress DC

        # Step 3 — radial profile on whitened residual
        mask   = hard_mask(H, W, beta)
        masked = amp_white * mask
        r_prof = radial_profile(masked, n_bins=n_bins)

        # Step 4 — band energy ratios
        r_max  = min(H, W) / 2.0
        ratios = _band_energy_ratios(amp_white, R, r_max)

        desc = np.concatenate([r_prof, ratios])   # (n_bins+6)-d

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

if __name__ == "__main__":

    # ══════════════════════════════════════════════════════════
    #  CONFIGURATION — edit these values directly
    # ══════════════════════════════════════════════════════════

    # ── Dataset ────────────────────────────────────────────────
    # "casiams" : 6 clients, one per spectral band
    #             (460nm, 630nm, 700nm, 850nm, 940nm, WHT)
    # "xjtu"    : 4 clients, one per (smartphone × lighting)
    #             (iPhone/Flash, iPhone/Nature, huawei/Flash, huawei/Nature)
    DATASET        = "casiams"

    # Override data path. None → uses CASIAMS_ROOT / XJTU_ROOT at top of file.
    DATA_ROOT      = None

    # ── Descriptor mode ────────────────────────────────────────
    # Determines what feature vector is extracted from each image's FFT.
    # Each point in the scatter plots represents one image's descriptor.
    # Better domain separation → domains are more distinguishable in frequency.
    #
    # "radial" (recommended — best PCA separation in experiments):
    #   1. Compute |FFT(image)|, fftshift so DC is at centre
    #   2. Apply log(1+amp) to compress dynamic range
    #      (without log, the DC component at r=0 dominates everything)
    #   3. Apply hard circular mask (radius = BETA × IMG_SIDE)
    #   4. Average log-amplitude over N_BINS concentric rings
    #   → descriptor: N_BINS-d (default 64-d)
    #   Why it works: different wavelength sensors (460nm vs 940nm) have
    #   different spectral decay rates — the log-radial profile captures
    #   this rolloff, which is domain-specific but identity-independent.
    #
    # "raw" (baseline):
    #   1. Compute |FFT(image)|
    #   2. Apply hard circular mask
    #   3. Flatten to vector
    #   → descriptor: (2r)²-d, r = BETA×IMG_SIDE (very high-dim, ~16384-d)
    #   Problem: no log → DC dominates; identity structure bleeds in.
    #   Use only to confirm that "radial" actually improves over this.
    #
    # "sensorprint" (forensic-style sensor fingerprint):
    #   1. residual = image − GaussianBlur(image, σ=2)
    #      Removes low-freq palm content (identity structure).
    #      Leaves high-freq acquisition artifacts (sensor, lens, compression).
    #   2. amp_log = log(1 + |FFT(residual)|)
    #   3. Whiten: amp_white = amp_log / r^ALPHA
    #      Removes universal 1/f² natural-image prior so only device-specific
    #      deviations from that prior remain.
    #   4. radial_profile(amp_white × hard_mask) → N_BINS-d
    #   5. band_energy_ratios(amp_white) → 6-d
    #      (ratios between 5 frequency bands — normalisation-invariant)
    #   → concat: (N_BINS+6)-d = 70-d
    #   Best for devices/lighting (XJTU). Worse for CASIA-MS because
    #   the residual removes the low-freq spectral rolloff that IS the
    #   domain signal there.
    DESC_MODE      = "raw"

    # ── FFT parameters ─────────────────────────────────────────
    # BETA: radius of the hard circular mask as a fraction of IMG_SIDE.
    #   Larger BETA → more of the frequency spectrum included.
    #   BETA=0.1 → inner 10% of frequencies (very low-freq only)
    #   BETA=0.4 → inner 40% (low + mid frequencies) ← recommended
    #   BETA=0.5 → inner 50% = entire half-spectrum
    #   For "radial" mode, BETA affects which rings are included.
    #   For "raw" mode, BETA determines descriptor dimension.
    BETA           = 0.20

    # N_BINS: number of concentric rings for radial profile.
    #   More bins → finer frequency resolution but noisier at high radii.
    #   32  → coarse, fast, less discriminative
    #   64  → recommended balance
    #   128 → fine-grained but slower
    N_BINS         = 256

    # ALPHA: whitening exponent for "sensorprint" mode only.
    #   Natural images follow 1/f^alpha power law (alpha≈1 for amplitude).
    #   Dividing by r^ALPHA removes this prior, leaving device deviations.
    #   ALPHA=1.0 → standard amplitude whitening (recommended start)
    #   ALPHA=0.5 → gentler whitening (keeps more low-freq signal)
    #   ALPHA=1.5 → aggressive (emphasises high-freq device artifacts)
    ALPHA          = 1.0

    # ── Projection methods ─────────────────────────────────────
    # Which 2D projections to compute. All three are shown side-by-side.
    # "pca"   : fast linear projection. Good for showing global structure.
    #           Fan/wedge shape = one dominant axis. Circular = distributed.
    # "tsne"  : non-linear, preserves local neighbourhood. Best for seeing
    #           whether domains form tight clusters. Slow on large datasets
    #           (auto-subsampled to MAX_TSNE=2000 points if needed).
    # "umap"  : non-linear, preserves both local and global structure.
    #           Often best overall. Requires: pip install umap-learn
    METHODS        = ["pca", "tsne", "umap"]

    # Set False to skip UMAP even if umap-learn is installed.
    # Useful if UMAP is slow or causing issues.
    USE_UMAP       = True

    # ── Domain classifier ──────────────────────────────────────
    # Trains a domain classifier using 5-fold cross-validation.
    # Reports accuracy + per-class metrics + confusion matrix.
    # High accuracy → descriptor contains genuine domain information.
    # Compare across DESC_MODE values to find the most discriminative one.
    #
    # "svm"  : RBF kernel SVM (C=10, gamma=scale).
    #          Best for: small/medium datasets, nonlinear boundaries.
    #          Scales well to 64-d or 70-d descriptors.
    # "nn"   : MLP with 2 hidden layers (256→128 for high-dim, 128→64 otherwise).
    #          Better for: high-dimensional "raw" descriptors (~16384-d).
    #          Slower to train, more sensitive to feature scaling.
    # "both" : runs both classifiers and reports both accuracies.
    CLASSIFIER     = "nn"

    # ── Sampling ───────────────────────────────────────────────
    # MAX_PER_DOMAIN: subsample to this many images per domain.
    #   None → use all images (CASIA-MS: ~1200 per domain, 7243 total)
    #   100  → fast preview, 600 total — good for quick iteration
    #   For t-SNE/UMAP: always auto-subsampled to 2000 total regardless.
    MAX_PER_DOMAIN = None

    # Input image size. Should match the FL training pipeline (128 for CompNet).
    IMG_SIDE       = 128

    # Random seed for reproducibility (PCA, t-SNE, UMAP, classifier CV).
    SEED           = 42

    # ── Output ─────────────────────────────────────────────────
    # Directory where all figures are saved as PNG files.
    # Output filenames include dataset name, desc_mode, and beta for traceability.
    # Example: fft_scatter_casiams_radial_beta0.4.png
    OUT_DIR        = "./plots"

    # ══════════════════════════════════════════════════════════
    #  END OF CONFIGURATION
    # ══════════════════════════════════════════════════════════

    # resolve data root
    if DATA_ROOT is None:
        DATA_ROOT = CASIAMS_ROOT if DATASET == "casiams" else XJTU_ROOT

    # apply toggles
    methods = [m for m in METHODS if m != "umap" or USE_UMAP]

    os.makedirs(OUT_DIR, exist_ok=True)
    colors, labels_map = get_palette(DATASET)

    print(f"\n{'='*56}")
    print(f"  FFT Domain Visualisation — {DATASET.upper()}")
    print(f"  data_root      : {DATA_ROOT}")
    print(f"  desc_mode      : {DESC_MODE}"
          + (f"  alpha={ALPHA}" if DESC_MODE == "sensorprint" else ""))
    print(f"  beta           : {BETA}   n_bins={N_BINS}")
    print(f"  classifier     : {CLASSIFIER}")
    print(f"  img_side       : {IMG_SIDE}")
    print(f"  max_per_domain : {MAX_PER_DOMAIN or 'all'}")
    print(f"  methods        : {methods}")
    print(f"  out_dir        : {OUT_DIR}")
    print(f"{'='*56}\n")

    # ── 1. Parse ───────────────────────────────────────────────
    print(f"Parsing {DATASET.upper()} …")
    if DATASET == "casiams":
        data = parse_casia_ms(DATA_ROOT)
    else:
        data = parse_xjtu(DATA_ROOT)

    domains = (sorted(data.keys()) if DATASET == "casiams"
               else [f"{d}/{c}" for d, c in XJTU_VARIATIONS
                     if f"{d}/{c}" in data])
    for sp in domains:
        n_ids = len(data[sp])
        n_img = sum(len(v) for v in data[sp].values())
        print(f"  {sp:>18}  IDs={n_ids}  images={n_img}")

    # ── 2. Extract descriptors ─────────────────────────────────
    print(f"\nExtracting FFT descriptors (β={BETA}) …")
    descriptors       = []
    domain_labels_arr = []
    mean_amps         = {}
    mean_descs        = {}

    for sp in domains:
        paths = [p for paths in data[sp].values() for p in paths]
        if MAX_PER_DOMAIN is not None:
            rng   = np.random.RandomState(SEED)
            paths = list(rng.choice(paths,
                                     min(MAX_PER_DOMAIN, len(paths)),
                                     replace=False))
        sp_descs, sp_amps = [], []
        for path in paths:
            try:
                amp, desc = extract_descriptor(
                path, IMG_SIDE, BETA,
                n_bins=N_BINS, mode=DESC_MODE,
                alpha=ALPHA)
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

    tag = f"{DATASET}_{DESC_MODE}_beta{BETA}"

    # ── 3. Scatter projections ─────────────────────────────────
    print("\nComputing projections …")
    plot_projections(
        descriptors, domain_labels_arr, domains, methods,
        colors, labels_map, DATASET, BETA, len(descriptors),
        out_path=os.path.join(OUT_DIR, f"fft_scatter_{tag}.png"),
        seed=SEED)

    # ── 4. Mean amplitude heatmaps ────────────────────────────
    print("\nPlotting mean amplitude heatmaps …")
    plot_mean_heatmaps(
        mean_amps, domains, IMG_SIDE, BETA,
        colors, labels_map,
        out_path=os.path.join(OUT_DIR, f"fft_heatmaps_{tag}.png"))

    # ── 5. Pairwise distance matrix ────────────────────────────
    print("Plotting pairwise distance matrix …")
    plot_distance_matrix(
        mean_descs, domains, labels_map,
        out_path=os.path.join(OUT_DIR, f"fft_distances_{tag}.png"))

    # ── 6. Radial profile curves ───────────────────────────────
    if DESC_MODE == "radial":
        print("Plotting radial spectral profiles …")
        plot_radial_profiles(
            mean_descs, domains, colors, labels_map, BETA,
            out_path=os.path.join(OUT_DIR, f"fft_radial_{tag}.png"))

    # ── 7. Domain classifier ───────────────────────────────────
    clf_names = (["svm", "nn"] if CLASSIFIER == "both"
                 else [CLASSIFIER])
    print(f"\nTraining domain classifier(s): {clf_names} …")
    results = {}
    for clf in clf_names:
        cm_path = os.path.join(
            OUT_DIR,
            f"clf_confusion_{clf}_{DESC_MODE}_{tag}.png")
        acc, _ = classify_domains(
            descriptors, domain_labels_arr, domains,
            classifier  = clf,
            desc_mode   = DESC_MODE,
            out_path    = cm_path,
            seed        = SEED)
        results[clf] = acc

    print(f"\n  ── Classifier Summary ──────────────────────────")
    print(f"  Dataset    : {DATASET.upper()}")
    print(f"  Descriptor : {DESC_MODE}  ({len(descriptors[0])}-d)")
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

    print(f"\nDone. Figures saved to: {OUT_DIR}/")


import os
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
