"""
pseudo_labels.py — Cross-expert pseudo-label supervision.

v5: Shared expert as teacher candidate + teacher composition tracking.
    Voters: backbone, shared-expert-only, previous domain experts (FDD-gated).
"""

import torch
import torch.nn.functional as F
from collections import Counter


@torch.no_grad()
def get_teacher_signals(model, images, active_domain, cfg,
                        fdd_distances=None):
    """
    Voters:
      - backbone (always)
      - shared expert only (backbone + shared, no domain expert)
      - previous domain experts (FDD distance gated)

    Returns: pseudo_labels, pl_mask, teacher_probs, kd_mask, stats
    """
    B = images.shape[0]
    device = images.device
    threshold = cfg.pl_threshold
    agreement_ratio = cfg.pl_agreement
    T = cfg.kd_temperature
    include_thresh = cfg.fdd_include_threshold

    voter_names = []
    voter_dists = []       # FDD distance for each voter (None for backbone/shared)
    all_preds = []
    all_confs = []
    all_soft = []

    # ─── Voter 0: frozen backbone ─────────────────────────────────────
    bb_logits = model.backbone(images)
    bb_probs = F.softmax(bb_logits, dim=-1)
    bb_conf, bb_pred = bb_probs.max(dim=-1)
    bb_soft = F.softmax(bb_logits / T, dim=-1)

    all_preds.append(bb_pred)
    all_confs.append(bb_conf)
    all_soft.append(bb_soft)
    voter_names.append("bb")
    voter_dists.append(None)

    # ─── Voter 1: shared expert only (backbone + shared, no domain) ───
    # Set active_domain to -1 → domain bypass = 0, only shared contributes
    saved_domain = active_domain
    model.set_active_domain(-1)
    shared_logits = model(images)
    shared_probs = F.softmax(shared_logits, dim=-1)
    shared_conf, shared_pred = shared_probs.max(dim=-1)
    shared_soft = F.softmax(shared_logits / T, dim=-1)
    model.set_active_domain(saved_domain)

    all_preds.append(shared_pred)
    all_confs.append(shared_conf)
    all_soft.append(shared_soft)
    voter_names.append("sh")
    voter_dists.append(None)

    # ─── Voters 2+: previous domain experts (FDD distance gated) ──────
    included_experts = []
    excluded_experts = []

    for dom_id in range(active_domain):
        if fdd_distances is not None and dom_id in fdd_distances:
            dist = fdd_distances[dom_id]
            if dist > include_thresh:
                excluded_experts.append((dom_id, dist))
                continue
            included_experts.append((dom_id, dist))
        else:
            included_experts.append((dom_id, -1.0))

        model.set_active_domain(dom_id)
        exp_logits = model(images)
        exp_probs = F.softmax(exp_logits, dim=-1)
        exp_conf, exp_pred = exp_probs.max(dim=-1)
        exp_soft = F.softmax(exp_logits / T, dim=-1)

        all_preds.append(exp_pred)
        all_confs.append(exp_conf)
        all_soft.append(exp_soft)
        voter_names.append(f"e{dom_id}")
        voter_dists.append(fdd_distances[dom_id] if fdd_distances and dom_id in fdd_distances else -1.0)

    model.set_active_domain(saved_domain)

    num_voters = len(all_preds)
    preds_stack = torch.stack(all_preds, dim=0)
    confs_stack = torch.stack(all_confs, dim=0)
    soft_stack = torch.stack(all_soft, dim=0)

    # ─── Group consensus ──────────────────────────────────────────────
    pseudo_labels = torch.zeros(B, dtype=torch.long, device=device)
    pl_mask = torch.zeros(B, dtype=torch.bool, device=device)
    teacher_probs = torch.zeros(B, bb_probs.shape[-1], device=device)
    teacher_conf_per_sample = torch.zeros(B, device=device)
    min_agree = max(1, int(num_voters * agreement_ratio))

    for i in range(B):
        sample_preds = preds_stack[:, i]
        sample_confs = confs_stack[:, i]
        sample_soft = soft_stack[:, i, :]

        confident = sample_confs > threshold
        n_confident = confident.sum().item()

        if n_confident == 0:
            teacher_probs[i] = soft_stack[0, i, :]
            teacher_conf_per_sample[i] = sample_confs[0]
            continue

        confident_preds = sample_preds[confident].cpu().tolist()
        counts = Counter(confident_preds)
        majority_class, majority_count = counts.most_common(1)[0]

        agrees = (sample_preds == majority_class)
        in_group = confident & agrees
        n_in_group = in_group.sum().item()

        if n_in_group >= min_agree:
            pseudo_labels[i] = majority_class
            pl_mask[i] = True

        if n_in_group > 0:
            w = sample_confs[in_group]
            w = w / w.sum()
            teacher_probs[i] = (w[:, None] * sample_soft[in_group]).sum(dim=0)
            teacher_conf_per_sample[i] = sample_confs[in_group].mean()
        else:
            teacher_probs[i] = soft_stack[0, i, :]
            teacher_conf_per_sample[i] = sample_confs[0]

    kd_mask = teacher_conf_per_sample > threshold
    teacher_pred = teacher_probs.argmax(dim=-1)

    # ─── Teacher composition string ───────────────────────────────────
    # e.g. "bb+sh+e1" or "bb+sh" (backbone + shared + expert 1)
    teach_str = "+".join(voter_names)

    # distance string for included experts: "-- / -- / 0.82"
    dist_parts = []
    for name, d in zip(voter_names, voter_dists):
        if d is None:
            dist_parts.append("--")
        else:
            dist_parts.append(f"{d:.2f}")
    dist_str = "/".join(dist_parts)

    stats = {
        "num_voters": num_voters,
        "num_agreed": pl_mask.sum().item(),
        "agreement_rate": pl_mask.float().mean().item() * 100,
        "teacher_conf": teacher_conf_per_sample.mean().item(),
        "experts_excluded": len(excluded_experts),
        "experts_included": len(included_experts),
        "kd_samples": kd_mask.sum().item(),
        "bb_pred": bb_pred,
        "teacher_pred": teacher_pred,
        "teach_str": teach_str,
        "dist_str": dist_str,
        "voter_names": voter_names,
        "voter_dists": voter_dists,
        "excluded_details": excluded_experts,
    }

    return pseudo_labels, pl_mask, teacher_probs, kd_mask, stats


def compute_pl_kd_loss(logits, pseudo_labels, pl_mask, teacher_probs, kd_mask, cfg):
    T = cfg.kd_temperature

    if cfg.pl_lambda > 0 and pl_mask.sum() > 0:
        pl_loss = F.cross_entropy(logits[pl_mask], pseudo_labels[pl_mask])
    else:
        pl_loss = torch.tensor(0.0, device=logits.device)

    if cfg.kd_lambda > 0 and kd_mask.sum() > 0:
        student_log_probs = F.log_softmax(logits[kd_mask] / T, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs[kd_mask],
                           reduction='batchmean') * (T * T)
    else:
        kd_loss = torch.tensor(0.0, device=logits.device)

    return pl_loss, kd_loss, {
        "pl_loss": pl_loss.item(), "kd_loss": kd_loss.item(),
        "pl_samples": pl_mask.sum().item(), "kd_samples": kd_mask.sum().item()}
