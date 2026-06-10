"""
pseudo_labels.py — Cross-expert pseudo-label supervision and
knowledge distillation for continual test-time adaptation.

Provides two supervision signals for domain expert k:

  1. Hard pseudo-labels (PL): Cross-entropy loss on samples where
     the teacher ensemble unanimously agrees with high confidence.
     Teachers = frozen backbone + experts 0..k-1.

  2. Soft knowledge distillation (KD): KL divergence from the current
     expert's distribution to the teacher ensemble's soft distribution.
     Teacher distribution = confidence-weighted mean of all voters.

For domain 0 (first expert):
  - Only the frozen backbone is available as teacher.
  - PL: samples where backbone confidence > threshold.
  - KD: distill from backbone's soft predictions.

For domain k > 0:
  - Teachers: backbone + expert_0 + expert_1 + ... + expert_{k-1}
  - Each teacher runs through the full model (backbone + their expert).
  - PL: samples where ≥ pl_agreement fraction of teachers agree.
  - KD: confidence-weighted average of all teacher distributions.
"""

import torch
import torch.nn.functional as F
from collections import Counter


@torch.no_grad()
def get_teacher_signals(model, images, active_domain, cfg):
    """
    Compute hard pseudo-labels and soft teacher distribution for the
    current domain's expert, using backbone + all previous experts.

    Args:
        model:          ExpertViT model (with set_active_domain)
        images:         [B, C, H, W] input batch
        active_domain:  current domain index (int)
        cfg:            config with pl_threshold, pl_agreement, kd_temperature

    Returns:
        pseudo_labels:  [B] int64 — agreed class per sample (valid where mask=True)
        pl_mask:        [B] bool — True for samples with consensus agreement
        teacher_probs:  [B, num_classes] float — soft teacher distribution for KD
        stats:          dict with diagnostic info (num_voters, num_agreed, etc.)
    """
    B = images.shape[0]
    device = images.device
    threshold = cfg.pl_threshold
    agreement_ratio = cfg.pl_agreement

    # ─── Collect predictions from all voters ──────────────────────────

    all_preds = []      # [num_voters, B] — hard predictions
    all_confs = []      # [num_voters, B] — max confidence per sample
    all_probs = []      # [num_voters, B, C] — full soft distributions
    voter_names = []    # for diagnostics

    # Voter 0: frozen backbone (no experts)
    bb_logits = model.backbone(images)
    bb_probs = F.softmax(bb_logits, dim=-1)
    bb_conf, bb_pred = bb_probs.max(dim=-1)

    all_preds.append(bb_pred)
    all_confs.append(bb_conf)
    all_probs.append(bb_probs)
    voter_names.append("backbone")

    # Voters 1..k: previous domain experts (each with shared expert)
    # We temporarily switch the active domain to get each expert's output
    for dom_id in range(active_domain):
        model.set_active_domain(dom_id)
        expert_logits = model(images)
        expert_probs = F.softmax(expert_logits, dim=-1)
        expert_conf, expert_pred = expert_probs.max(dim=-1)

        all_preds.append(expert_pred)
        all_confs.append(expert_conf)
        all_probs.append(expert_probs)
        voter_names.append(f"expert_{dom_id}")

    # Restore active domain
    model.set_active_domain(active_domain)

    num_voters = len(all_preds)
    preds_stack = torch.stack(all_preds, dim=0)   # [V, B]
    confs_stack = torch.stack(all_confs, dim=0)    # [V, B]
    probs_stack = torch.stack(all_probs, dim=0)    # [V, B, C]

    # ─── Hard pseudo-labels: multi-model agreement (Criterion B) ──────

    # For each sample, find the most common prediction among voters
    # and check if enough voters agree AND are confident
    pseudo_labels = torch.zeros(B, dtype=torch.long, device=device)
    pl_mask = torch.zeros(B, dtype=torch.bool, device=device)
    min_agree = max(1, int(num_voters * agreement_ratio))

    for i in range(B):
        sample_preds = preds_stack[:, i]            # [V]
        sample_confs = confs_stack[:, i]            # [V]

        # only count votes from confident voters
        confident_mask = sample_confs > threshold   # [V]
        if confident_mask.sum() < min_agree:
            continue

        confident_preds = sample_preds[confident_mask]

        # find majority class among confident voters
        pred_list = confident_preds.cpu().tolist()
        counts = Counter(pred_list)
        majority_class, majority_count = counts.most_common(1)[0]

        # check if enough confident voters agree
        if majority_count >= min_agree:
            pseudo_labels[i] = majority_class
            pl_mask[i] = True

    # ─── Soft teacher distribution: confidence-weighted ensemble ──────
    #
    # Each voter's weight = mean confidence on this batch
    # Higher confidence voter contributes more to the soft target
    #
    # teacher_probs[i] = Σ_v  w_v · softmax(logits_v[i] / T)
    #                    ─────────────────────────────────────
    #                              Σ_v w_v

    T = cfg.kd_temperature

    # compute temperature-scaled distributions
    # We need to recompute with temperature since probs_stack used T=1
    if T != 1.0:
        # re-collect with temperature (backbone)
        soft_probs_list = [F.softmax(bb_logits / T, dim=-1)]

        # re-collect previous experts with temperature
        for dom_id in range(active_domain):
            model.set_active_domain(dom_id)
            expert_logits_t = model(images)
            soft_probs_list.append(F.softmax(expert_logits_t / T, dim=-1))
        model.set_active_domain(active_domain)

        soft_probs_stack = torch.stack(soft_probs_list, dim=0)  # [V, B, C]
    else:
        soft_probs_stack = probs_stack

    # per-voter confidence weight = mean max confidence across batch
    voter_weights = confs_stack.mean(dim=1)         # [V]
    voter_weights = voter_weights / voter_weights.sum()  # normalize

    # weighted average of soft distributions
    # [V, 1, 1] × [V, B, C] → sum over V → [B, C]
    teacher_probs = (voter_weights[:, None, None] * soft_probs_stack).sum(dim=0)

    # ─── Diagnostics ──────────────────────────────────────────────────
    stats = {
        "num_voters": num_voters,
        "voter_names": voter_names,
        "num_agreed": pl_mask.sum().item(),
        "agreement_rate": pl_mask.float().mean().item() * 100,
        "voter_weights": {name: w.item()
                          for name, w in zip(voter_names, voter_weights)},
    }

    return pseudo_labels, pl_mask, teacher_probs, stats


def compute_pl_kd_loss(logits, pseudo_labels, pl_mask, teacher_probs, cfg):
    """
    Compute pseudo-label cross-entropy + knowledge distillation losses.

    Args:
        logits:         [B, C] current expert's raw logits
        pseudo_labels:  [B] hard pseudo-labels from consensus
        pl_mask:        [B] bool — which samples have valid pseudo-labels
        teacher_probs:  [B, C] soft teacher distribution
        cfg:            config with pl_lambda, kd_lambda, kd_temperature

    Returns:
        pl_loss:    scalar — cross-entropy on agreed samples (0 if none)
        kd_loss:    scalar — KL divergence from student to teacher
        stats:      dict with loss values for logging
    """
    T = cfg.kd_temperature

    # ─── Hard pseudo-label loss (cross-entropy on agreed samples) ─────
    if cfg.pl_lambda > 0 and pl_mask.sum() > 0:
        pl_loss = F.cross_entropy(logits[pl_mask], pseudo_labels[pl_mask])
    else:
        pl_loss = torch.tensor(0.0, device=logits.device)

    # ─── Soft knowledge distillation loss (KL divergence) ─────────────
    #
    # Standard KD formulation (Hinton et al., 2015):
    #   L_KD = T² × KL(softmax(teacher/T) || softmax(student/T))
    #
    # The T² scaling ensures gradient magnitudes are independent of T.

    if cfg.kd_lambda > 0:
        student_log_probs = F.log_softmax(logits / T, dim=-1)
        # teacher_probs is already temperature-scaled from get_teacher_signals
        kd_loss = F.kl_div(student_log_probs, teacher_probs,
                           reduction='batchmean') * (T * T)
    else:
        kd_loss = torch.tensor(0.0, device=logits.device)

    stats = {
        "pl_loss": pl_loss.item(),
        "kd_loss": kd_loss.item(),
        "pl_samples": pl_mask.sum().item(),
    }

    return pl_loss, kd_loss, stats
