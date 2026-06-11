"""
pseudo_labels.py — Cross-expert pseudo-label supervision and
knowledge distillation for continual test-time adaptation.

v4: FDD distance-based teacher selection.
    - Previous experts included only if their FDD domain is close
      (distance < fdd_include_threshold)
    - Group consensus among included voters
    - KD filtered to teacher-confident samples
"""

import torch
import torch.nn.functional as F
from collections import Counter


@torch.no_grad()
def get_teacher_signals(model, images, active_domain, cfg,
                        fdd_distances=None):
    """
    Compute hard pseudo-labels and soft teacher distribution.

    FDD distance gating: only include expert j if
    fdd_distances[j] < cfg.fdd_include_threshold.

    Args:
        model:          ExpertViT model
        images:         [B, C, H, W] input batch
        active_domain:  current domain index (int)
        cfg:            config
        fdd_distances:  dict {domain_id: distance} from FDD, or None

    Returns:
        pseudo_labels, pl_mask, teacher_probs, kd_mask, stats
    """
    B = images.shape[0]
    device = images.device
    threshold = cfg.pl_threshold
    agreement_ratio = cfg.pl_agreement
    T = cfg.kd_temperature
    include_thresh = cfg.fdd_include_threshold

    # ─── Voter 0: frozen backbone (always included) ───────────────────
    bb_logits = model.backbone(images)
    bb_probs = F.softmax(bb_logits, dim=-1)
    bb_conf, bb_pred = bb_probs.max(dim=-1)
    bb_soft = F.softmax(bb_logits / T, dim=-1)

    all_logits = [bb_logits]
    all_preds = [bb_pred]
    all_confs = [bb_conf]
    all_soft = [bb_soft]
    voter_names = ["backbone"]
    included_experts = []
    excluded_experts = []

    # ─── Previous experts: include only if FDD distance is close ──────
    for dom_id in range(active_domain):
        # FDD distance gating
        if fdd_distances is not None and dom_id in fdd_distances:
            dist = fdd_distances[dom_id]
            if dist > include_thresh:
                excluded_experts.append((dom_id, dist))
                continue  # too far in frequency space → skip
            included_experts.append((dom_id, dist))
        else:
            # no distance info → include (conservative)
            included_experts.append((dom_id, -1.0))

        model.set_active_domain(dom_id)
        expert_logits = model(images)
        expert_probs = F.softmax(expert_logits, dim=-1)
        expert_conf, expert_pred = expert_probs.max(dim=-1)
        expert_soft = F.softmax(expert_logits / T, dim=-1)

        all_logits.append(expert_logits)
        all_preds.append(expert_pred)
        all_confs.append(expert_conf)
        all_soft.append(expert_soft)
        voter_names.append(f"expert_{dom_id}")

    # restore active domain
    model.set_active_domain(active_domain)

    num_voters = len(all_preds)

    # ─── Single voter (backbone only) — fast path ────────────────────
    if num_voters == 1:
        pl_mask = bb_conf > threshold
        pseudo_labels = bb_pred
        kd_mask = bb_conf > threshold

        stats = {
            "num_voters": 1,
            "num_agreed": pl_mask.sum().item(),
            "agreement_rate": pl_mask.float().mean().item() * 100,
            "teacher_conf": bb_conf.mean().item(),
            "experts_excluded": len(excluded_experts),
            "experts_included": 0,
            "kd_samples": kd_mask.sum().item(),
            "bb_pred": bb_pred,
            "teacher_pred": bb_pred,
            "excluded_details": excluded_experts,
        }
        return pseudo_labels, pl_mask, bb_soft, kd_mask, stats

    # ─── Multiple voters: group consensus ─────────────────────────────
    preds_stack = torch.stack(all_preds, dim=0)    # [V, B]
    confs_stack = torch.stack(all_confs, dim=0)     # [V, B]
    soft_stack = torch.stack(all_soft, dim=0)       # [V, B, C]

    pseudo_labels = torch.zeros(B, dtype=torch.long, device=device)
    pl_mask = torch.zeros(B, dtype=torch.bool, device=device)
    teacher_probs = torch.zeros(B, bb_probs.shape[-1], device=device)
    teacher_conf_per_sample = torch.zeros(B, device=device)
    min_agree = max(1, int(num_voters * agreement_ratio))

    total_consensus_excluded = 0

    for i in range(B):
        sample_preds = preds_stack[:, i]
        sample_confs = confs_stack[:, i]
        sample_soft = soft_stack[:, i, :]

        # confident voters
        confident = sample_confs > threshold
        n_confident = confident.sum().item()

        if n_confident == 0:
            teacher_probs[i] = soft_stack[0, i, :]
            teacher_conf_per_sample[i] = sample_confs[0]
            continue

        # majority class among confident voters
        confident_preds = sample_preds[confident].cpu().tolist()
        counts = Counter(confident_preds)
        majority_class, majority_count = counts.most_common(1)[0]

        # in-group: confident AND agrees with majority
        agrees = (sample_preds == majority_class)
        in_group = confident & agrees
        n_in_group = in_group.sum().item()
        total_consensus_excluded += (n_confident - n_in_group)

        # hard pseudo-label
        if n_in_group >= min_agree:
            pseudo_labels[i] = majority_class
            pl_mask[i] = True

        # soft teacher from in-group
        if n_in_group > 0:
            w = sample_confs[in_group]
            w = w / w.sum()
            teacher_probs[i] = (w[:, None] * sample_soft[in_group]).sum(dim=0)
            teacher_conf_per_sample[i] = sample_confs[in_group].mean()
        else:
            teacher_probs[i] = soft_stack[0, i, :]
            teacher_conf_per_sample[i] = sample_confs[0]

    # KD mask
    kd_mask = teacher_conf_per_sample > threshold
    teacher_pred = teacher_probs.argmax(dim=-1)

    stats = {
        "num_voters": num_voters,
        "num_agreed": pl_mask.sum().item(),
        "agreement_rate": pl_mask.float().mean().item() * 100,
        "teacher_conf": teacher_conf_per_sample.mean().item(),
        "experts_excluded": len(excluded_experts),
        "experts_included": len(included_experts),
        "consensus_excluded_per_sample": total_consensus_excluded / B,
        "kd_samples": kd_mask.sum().item(),
        "bb_pred": bb_pred,
        "teacher_pred": teacher_pred,
        "excluded_details": excluded_experts,
    }

    return pseudo_labels, pl_mask, teacher_probs, kd_mask, stats


def compute_pl_kd_loss(logits, pseudo_labels, pl_mask, teacher_probs, kd_mask, cfg):
    """
    Hard PL cross-entropy (on agreed samples) +
    Filtered soft KD (on teacher-confident samples).
    """
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

    stats = {
        "pl_loss": pl_loss.item(),
        "kd_loss": kd_loss.item(),
        "pl_samples": pl_mask.sum().item(),
        "kd_samples": kd_mask.sum().item(),
    }
    return pl_loss, kd_loss, stats
