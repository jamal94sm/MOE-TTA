"""
pseudo_labels.py — Cross-expert pseudo-label supervision and
knowledge distillation for continual test-time adaptation.

v3: Group consensus filtering — the majority group determines the
    teacher for each sample. Any voter (including backbone) that
    disagrees with the majority is excluded per-sample.
    
    Teacher ensemble for sample i:
      1. All voters (backbone + experts 0..k-1) predict
      2. Only confident voters (max(softmax) > threshold) participate
      3. Find majority class among confident voters
      4. Exclude voters that disagree with the majority
      5. Teacher = confidence-weighted mean of remaining voters' soft predictions
"""

import torch
import torch.nn.functional as F
from collections import Counter


@torch.no_grad()
def get_teacher_signals(model, images, active_domain, cfg):
    """
    Compute hard pseudo-labels and soft teacher distribution using
    group consensus: the majority determines the teacher, not the backbone.

    Args:
        model:          ExpertViT model
        images:         [B, C, H, W] input batch
        active_domain:  current domain index (int)
        cfg:            config with pl_threshold, pl_agreement, kd_temperature

    Returns:
        pseudo_labels:  [B] hard labels from consensus
        pl_mask:        [B] bool — samples with strong agreement
        teacher_probs:  [B, C] soft teacher distribution (filtered)
        kd_mask:        [B] bool — samples where teacher is confident for KD
        stats:          dict with diagnostics
    """
    B = images.shape[0]
    device = images.device
    threshold = cfg.pl_threshold
    agreement_ratio = cfg.pl_agreement
    T = cfg.kd_temperature

    # ─── Collect all voter predictions ────────────────────────────────

    all_logits = []     # [V][B, C]
    all_preds = []      # [V][B]
    all_confs = []      # [V][B]
    voter_names = []

    # Voter 0: frozen backbone
    bb_logits = model.backbone(images)
    bb_probs = F.softmax(bb_logits, dim=-1)
    bb_conf, bb_pred = bb_probs.max(dim=-1)

    all_logits.append(bb_logits)
    all_preds.append(bb_pred)
    all_confs.append(bb_conf)
    voter_names.append("backbone")

    # Voters 1..k: previous domain experts
    for dom_id in range(active_domain):
        model.set_active_domain(dom_id)
        expert_logits = model(images)
        expert_probs = F.softmax(expert_logits, dim=-1)
        expert_conf, expert_pred = expert_probs.max(dim=-1)

        all_logits.append(expert_logits)
        all_preds.append(expert_pred)
        all_confs.append(expert_conf)
        voter_names.append(f"expert_{dom_id}")

    # restore active domain
    model.set_active_domain(active_domain)

    num_voters = len(all_preds)
    preds_stack = torch.stack(all_preds, dim=0)    # [V, B]
    confs_stack = torch.stack(all_confs, dim=0)     # [V, B]
    logits_stack = torch.stack(all_logits, dim=0)   # [V, B, C]

    # temperature-scaled soft distributions
    soft_stack = F.softmax(logits_stack / T, dim=-1)  # [V, B, C]

    # ─── Per-sample group consensus filtering ─────────────────────────
    #
    # For each sample:
    #   1. Find confident voters (conf > threshold)
    #   2. Among confident voters, find the majority class
    #   3. Voters agreeing with majority = "in-group" (trusted)
    #   4. Voters disagreeing = "excluded"
    #   5. Teacher = confidence-weighted average of in-group soft predictions

    pseudo_labels = torch.zeros(B, dtype=torch.long, device=device)
    pl_mask = torch.zeros(B, dtype=torch.bool, device=device)
    teacher_probs = torch.zeros(B, soft_stack.shape[-1], device=device)
    teacher_conf_per_sample = torch.zeros(B, device=device)
    min_agree = max(1, int(num_voters * agreement_ratio))

    total_excluded = 0  # track how many voter-sample pairs are excluded

    for i in range(B):
        sample_preds = preds_stack[:, i]       # [V]
        sample_confs = confs_stack[:, i]        # [V]
        sample_soft = soft_stack[:, i, :]       # [V, C]

        # step 1: filter confident voters
        confident = sample_confs > threshold    # [V]
        n_confident = confident.sum().item()

        if n_confident == 0:
            # no confident voter → use backbone as fallback
            teacher_probs[i] = soft_stack[0, i, :]
            teacher_conf_per_sample[i] = sample_confs[0]
            continue

        # step 2: find majority class among confident voters
        confident_preds = sample_preds[confident].cpu().tolist()
        counts = Counter(confident_preds)
        majority_class, majority_count = counts.most_common(1)[0]

        # step 3: in-group = confident AND agrees with majority
        agrees_with_majority = (sample_preds == majority_class)  # [V]
        in_group = confident & agrees_with_majority               # [V]
        n_in_group = in_group.sum().item()
        total_excluded += (confident.sum().item() - n_in_group)

        # step 4: hard pseudo-label (if enough voters agree)
        if n_in_group >= min_agree:
            pseudo_labels[i] = majority_class
            pl_mask[i] = True

        # step 5: soft teacher from in-group voters
        if n_in_group > 0:
            w = sample_confs[in_group]              # [n_in_group]
            w = w / w.sum()                         # normalize
            in_group_soft = sample_soft[in_group]   # [n_in_group, C]
            teacher_probs[i] = (w[:, None] * in_group_soft).sum(dim=0)
            teacher_conf_per_sample[i] = sample_confs[in_group].mean()
        else:
            # fallback: backbone only
            teacher_probs[i] = soft_stack[0, i, :]
            teacher_conf_per_sample[i] = sample_confs[0]

    # ─── KD mask: only distill where filtered teacher is confident ────
    kd_mask = teacher_conf_per_sample > threshold

    # ─── Teacher ensemble prediction (for accuracy diagnostics) ───────
    teacher_pred = teacher_probs.argmax(dim=-1)

    # ─── Stats ────────────────────────────────────────────────────────
    avg_excluded = total_excluded / B if num_voters > 1 else 0

    stats = {
        "num_voters": num_voters,
        "num_agreed": pl_mask.sum().item(),
        "agreement_rate": pl_mask.float().mean().item() * 100,
        "teacher_conf": teacher_conf_per_sample.mean().item(),
        "experts_excluded": avg_excluded,
        "kd_samples": kd_mask.sum().item(),
        "bb_pred": bb_pred,
        "teacher_pred": teacher_pred,
    }

    return pseudo_labels, pl_mask, teacher_probs, kd_mask, stats


def compute_pl_kd_loss(logits, pseudo_labels, pl_mask, teacher_probs, kd_mask, cfg):
    """
    Compute PL cross-entropy + filtered KD loss.
    Both are applied only on their respective masked samples.
    """
    T = cfg.kd_temperature

    # ─── Hard PL loss (only on agreed samples) ────────────────────────
    if cfg.pl_lambda > 0 and pl_mask.sum() > 0:
        pl_loss = F.cross_entropy(logits[pl_mask], pseudo_labels[pl_mask])
    else:
        pl_loss = torch.tensor(0.0, device=logits.device)

    # ─── Filtered KD loss (only on teacher-confident samples) ─────────
    if cfg.kd_lambda > 0 and kd_mask.sum() > 0:
        student_log_probs = F.log_softmax(logits[kd_mask] / T, dim=-1)
        teacher_target = teacher_probs[kd_mask]
        kd_loss = F.kl_div(student_log_probs, teacher_target,
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
