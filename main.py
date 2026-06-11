"""
main.py — Complete CTTA adaptation loop for the AAAI 2026 paper.

v4: Group consensus teacher + warmup phase for new experts.
    - Teacher = majority-group consensus (not backbone-as-reference)
    - New experts warm up with entropy+diversity only before PL/KD kicks in
    - KD filtered to teacher-confident samples only
"""

import os, sys, math, time, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from collections import Counter

from config import get_cfg
from model import build_model
from fdd import FrequencyDomainDiscriminator
from datasets import get_domain_sequence
from pseudo_labels import get_teacher_signals, compute_pl_kd_loss


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def filtered_entropy_loss(logits, threshold, entropy_floor=0.0, div_lambda=0.0):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    mask = (entropy < threshold).float()
    if entropy_floor > 0:
        mask = mask * (entropy > entropy_floor).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    ent_loss = (entropy * mask).sum() / mask.sum()
    if div_lambda > 0:
        batch_mean_prob = probs.mean(dim=0)
        log_C = math.log(probs.shape[-1])
        div_loss = (batch_mean_prob * torch.log(batch_mean_prob + 1e-8)).sum() + log_C
        return ent_loss + div_lambda * div_loss
    return ent_loss


def adapt(cfg):
    print(f"\n{'='*90}")
    print(f"  AAAI 2026: Shared & Domain Self-Adaptive Experts with FDD")
    print(f"  Dataset: {cfg.dataset} | LR: {cfg.lr} | BS: {cfg.batch_size}")
    print(f"{'='*90}\n")

    model = build_model(cfg)
    print(f"[Model] Backbone: {cfg.backbone}")
    print(f"[Model] Shared rank: {cfg.shared_rank}, "
          f"Domain rank: {cfg.domain_rank}, Experts/MoE: {cfg.num_experts_per_moe}")

    fdd = FrequencyDomainDiscriminator(
        freq_radius=cfg.fdd_freq_radius, threshold=cfg.fdd_threshold,
        shrinkage=cfg.fdd_shrinkage, init_var=cfg.fdd_init_var,
        diagonal=cfg.fdd_diagonal, device=cfg.device)
    print(f"[FDD] freq_radius={cfg.fdd_freq_radius}, "
          f"threshold={cfg.fdd_threshold}, diagonal={cfg.fdd_diagonal}")

    domain_sequence = get_domain_sequence(cfg)
    total_batches = sum(len(loader) for _, loader in domain_sequence)
    print(f"[Data] {len(domain_sequence)} domain segments, {total_batches} total batches")
    print(f"[Anti-collapse] entropy_floor={cfg.entropy_floor}, "
          f"stochastic_restore={cfg.stochastic_restore}, div_lambda={cfg.div_lambda}")
    print(f"[Optimizer] Constant LR={cfg.lr}, weight_decay={cfg.weight_decay}")
    if cfg.use_pseudo_labels:
        print(f"[PL/KD] ENABLED: pl_lambda={cfg.pl_lambda}, kd_lambda={cfg.kd_lambda}, "
              f"pl_threshold={cfg.pl_threshold}, pl_agreement={cfg.pl_agreement}, "
              f"kd_temperature={cfg.kd_temperature}, pl_warmup={cfg.pl_warmup}")
    else:
        print(f"[PL/KD] Disabled")

    # ── backbone baseline ──
    eval_bb = getattr(cfg, 'eval_backbone', False)
    baseline_errors = {}
    if eval_bb:
        print(f"\n[Baseline] Evaluating frozen backbone...")
        model.eval()
        with torch.no_grad():
            for dn, loader in domain_sequence:
                c = t = 0
                for imgs, labs in loader:
                    imgs, labs = imgs.to(cfg.device), labs.to(cfg.device)
                    p = model.backbone(imgs).argmax(1)
                    c += (p == labs).sum().item(); t += labs.shape[0]
                baseline_errors[dn] = 100.0 * (1 - c / t)
                print(f"  {dn:25s} → {baseline_errors[dn]:.1f}%")
        print(f"[Baseline] Mean: {np.mean(list(baseline_errors.values())):.1f}%\n")

    # ── stochastic restore state ──
    shared_init_state = {}
    if cfg.stochastic_restore > 0:
        for i, em in enumerate(model.expert_modules):
            for name, p in em.shared_moe.named_parameters():
                shared_init_state[f"{i}.{name}"] = p.data.clone()

    optimizer = None
    results = {}; domain_map = {}
    total_correct = total_samples = 0
    current_fdd_domain = -1; global_step = 0
    batches_since_new_domain = 0  # warmup counter

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # ── header ──
    hdr = (f"  {'bat':>5} │{'err%':>5} │{'H_mean':>6} │{'filt%':>5} │"
           f"{'top%':>4} │{'H_cor':>5} │{'H_wrg':>5}")
    if cfg.use_pseudo_labels:
        hdr += (f" │{'PL%':>4} │{'PLac':>4} │{'T_er':>4} │"
                f"{'KD%':>4} │{'excl':>4} │{'PLls':>6} │{'KDls':>6} │{'phase':>6}")

    for seg_idx, (domain_name, loader) in enumerate(domain_sequence):
        seg_correct = seg_total = 0; seg_loss_sum = 0.0
        n_batches = len(loader); t0 = time.time()
        all_errors = []; top_class_history = []
        updates_applied = updates_skipped = 0
        pl_total_agreed = pl_total_samples = 0
        pl_loss_sum = kd_loss_sum = 0.0
        pl_correct_sum = teacher_correct_sum = teacher_total = 0
        warmup_batches_used = 0

        print(f"\n{'─'*90}")
        print(f"  [{seg_idx+1}/{len(domain_sequence)}] {domain_name} "
              f"({len(loader.dataset)} samples, {n_batches} batches)")
        print(f"{'─'*90}")
        print(hdr)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(cfg.device); labels = labels.to(cfg.device)
            B = images.shape[0]

            # ─── FDD ───
            mean = imagenet_mean.to(cfg.device); std = imagenet_std.to(cfg.device)
            raw_images = images * std + mean
            fdd_domain_id, is_new = fdd.detect_domain(raw_images)

            if is_new:
                expert_id = model.expand_domain()
                model.set_active_domain(expert_id)
                current_fdd_domain = fdd_domain_id
                domain_map[fdd_domain_id] = [domain_name]
                batches_since_new_domain = 0  # reset warmup counter
                print(f"  >>> FDD: New domain {fdd_domain_id} → expert {expert_id} "
                      f"(warmup for {cfg.pl_warmup} batches)")
                if optimizer is None:
                    optimizer = Adam(model.get_trainable_params(),
                                    lr=cfg.lr, betas=(0.9, 0.999),
                                    weight_decay=cfg.weight_decay)
                else:
                    new_params = []
                    for em in model.expert_modules:
                        if expert_id < len(em.domain_moes):
                            new_params.extend(em.domain_moes[expert_id].parameters())
                    if new_params:
                        optimizer.add_param_group({
                            'params': new_params, 'lr': cfg.lr,
                            'betas': (0.9, 0.999), 'weight_decay': cfg.weight_decay})

            elif fdd_domain_id != current_fdd_domain:
                model.set_active_domain(fdd_domain_id)
                current_fdd_domain = fdd_domain_id
                if fdd_domain_id in domain_map:
                    domain_map[fdd_domain_id].append(domain_name)
                else:
                    domain_map[fdd_domain_id] = [domain_name]
                print(f"  >>> FDD: Matched domain {fdd_domain_id} → expert {fdd_domain_id}")

            # ─── Forward ───
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            preds = logits.argmax(dim=-1)

            # ─── Metrics ───
            correct = (preds == labels).sum().item()
            err = 100.0 * (1 - correct / B)
            all_errors.append(err); seg_correct += correct; seg_total += B

            correct_mask = (preds == labels)
            H_correct = entropy[correct_mask].mean().item() if correct_mask.any() else -1
            H_wrong = entropy[~correct_mask].mean().item() if (~correct_mask).any() else -1

            pred_counts = Counter(preds.cpu().tolist())
            top_pred, top_count = pred_counts.most_common(1)[0]
            top_pct = 100 * top_count / B
            top_class_history.append((top_pred, top_pct))

            filt_mask = entropy < cfg.entropy_threshold
            if cfg.entropy_floor > 0:
                filt_mask = filt_mask & (entropy > cfg.entropy_floor)
            pass_rate = filt_mask.float().mean().item() * 100

            # ─── Losses ───
            ent_loss = filtered_entropy_loss(logits, cfg.entropy_threshold,
                                              cfg.entropy_floor, cfg.div_lambda)

            batch_pl_loss = batch_kd_loss = 0.0
            batch_pl_rate = batch_pl_acc = batch_teacher_err = 0.0
            batch_kd_rate = batch_excl = 0.0

            # determine if we're in warmup phase
            in_warmup = (cfg.use_pseudo_labels and
                         batches_since_new_domain < cfg.pl_warmup)
            phase = "warmup" if in_warmup else "pl+kd"

            if cfg.use_pseudo_labels and not in_warmup:
                # ─── PL/KD active ───
                pseudo_labels_t, pl_mask, teacher_probs, kd_mask, pl_stats = \
                    get_teacher_signals(model, images, current_fdd_domain, cfg)

                pl_loss, kd_loss, loss_stats = compute_pl_kd_loss(
                    logits, pseudo_labels_t, pl_mask, teacher_probs, kd_mask, cfg)

                batch_pl_loss = loss_stats["pl_loss"]
                batch_kd_loss = loss_stats["kd_loss"]
                batch_pl_rate = pl_stats["agreement_rate"]
                batch_kd_rate = 100.0 * loss_stats["kd_samples"] / B
                batch_excl = pl_stats.get("experts_excluded", 0)

                # PL accuracy diagnostic (uses GT labels)
                if pl_mask.sum() > 0:
                    batch_pl_acc = (pseudo_labels_t[pl_mask] == labels[pl_mask]).float().mean().item() * 100
                    pl_correct_sum += (pseudo_labels_t[pl_mask] == labels[pl_mask]).sum().item()
                else:
                    batch_pl_acc = -1

                # teacher error diagnostic
                teacher_pred = pl_stats["teacher_pred"]
                t_correct = (teacher_pred == labels).sum().item()
                batch_teacher_err = 100.0 * (1 - t_correct / B)
                teacher_correct_sum += t_correct; teacher_total += B

                pl_total_agreed += loss_stats["pl_samples"]
                pl_total_samples += B
                pl_loss_sum += batch_pl_loss * B
                kd_loss_sum += batch_kd_loss * B

                # combine
                total_loss = ent_loss
                if cfg.pl_lambda > 0 and pl_loss.item() > 0:
                    total_loss = total_loss + cfg.pl_lambda * pl_loss
                if cfg.kd_lambda > 0 and kd_loss.item() > 0:
                    total_loss = total_loss + cfg.kd_lambda * kd_loss
            else:
                # warmup or PL disabled: entropy + diversity only
                total_loss = ent_loss
                if in_warmup:
                    warmup_batches_used += 1

            seg_loss_sum += total_loss.item() * B

            # ─── Update ───
            if optimizer is not None and total_loss.item() > 0:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                updates_applied += 1
                if cfg.stochastic_restore > 0:
                    with torch.no_grad():
                        for i, em_r in enumerate(model.expert_modules):
                            for name, p in em_r.shared_moe.named_parameters():
                                key = f"{i}.{name}"
                                if key in shared_init_state:
                                    m = torch.rand_like(p) < cfg.stochastic_restore
                                    p.data = torch.where(m, shared_init_state[key], p.data)
            else:
                updates_skipped += 1

            batches_since_new_domain += 1

            # ─── Print ───
            if batch_idx < 5 or batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                hc = f"{H_correct:.2f}" if H_correct >= 0 else " N/A"
                hw = f"{H_wrong:.2f}" if H_wrong >= 0 else " N/A"
                line = (f"  {batch_idx:5d} │{err:5.1f} │{entropy.mean().item():6.3f} │"
                        f"{pass_rate:5.1f} │{top_pct:4.0f} │{hc:>5s} │{hw:>5s}")
                if cfg.use_pseudo_labels:
                    pla = f"{batch_pl_acc:.0f}" if batch_pl_acc > 0 else " --"
                    if in_warmup:
                        line += (f" │  -- │  -- │  -- │  -- │  -- │   -- │   -- │"
                                 f" {'warm':>6s}")
                    else:
                        line += (f" │{batch_pl_rate:4.0f} │{pla:>4s} │"
                                 f"{batch_teacher_err:4.0f} │{batch_kd_rate:4.0f} │"
                                 f"{batch_excl:4.1f} │{batch_pl_loss:6.3f} │"
                                 f"{batch_kd_loss:6.3f} │ {'pl+kd':>5s}")
                print(line)

            total_correct += correct; total_samples += B; global_step += 1

        # ── domain summary ──
        seg_acc = seg_correct / max(seg_total, 1) * 100
        seg_err = 100.0 - seg_acc
        seg_loss_avg = seg_loss_sum / max(seg_total, 1)
        elapsed = time.time() - t0
        avg_err = np.mean(all_errors)

        results[domain_name] = {
            "error": seg_err, "accuracy": seg_acc, "loss": seg_loss_avg,
            "samples": seg_total, "fdd_domain": current_fdd_domain, "time": elapsed}

        q = len(all_errors) // 4
        first_q = np.mean(all_errors[:q]) if q > 0 else avg_err
        last_q = np.mean(all_errors[-q:]) if q > 0 else avg_err
        trend = last_q - first_q

        takeover_batch = None
        for b_idx, (cls, pct) in enumerate(top_class_history):
            if pct > 50: takeover_batch = b_idx; break

        baseline_err = baseline_errors.get(domain_name, None)
        print(f"\n  ┌── SUMMARY: {domain_name}")
        if baseline_err is not None:
            imp = baseline_err - avg_err
            print(f"  │ Backbone Error:  {baseline_err:.1f}%")
            print(f"  │ TTA Error:       {avg_err:.1f}%  "
                  f"(first25%={first_q:.1f}% → last25%={last_q:.1f}%, Δ={trend:+.1f}%)")
            print(f"  │ Improvement:     {'↓' if imp > 0 else '↑'} {abs(imp):.1f}%"
                  f"{'  ⚠ TTA HURTS' if imp < -1 else ''}")
        else:
            print(f"  │ Avg Error:       {avg_err:.1f}%  "
                  f"(first25%={first_q:.1f}% → last25%={last_q:.1f}%, Δ={trend:+.1f}%)")
        print(f"  │ Loss: {seg_loss_avg:.4f} | FDD: {fdd.num_domains} domains | "
              f"Updates: {updates_applied}/{updates_applied + updates_skipped} | "
              f"Time: {elapsed:.1f}s")
        if cfg.use_pseudo_labels:
            if pl_total_agreed > 0:
                pl_acc_avg = 100.0 * pl_correct_sum / pl_total_agreed
            else:
                pl_acc_avg = 0.0
            teacher_err_avg = 100.0 * (1 - teacher_correct_sum / max(teacher_total, 1)) if teacher_total > 0 else 0.0
            pl_avg_rate = 100.0 * pl_total_agreed / max(pl_total_samples, 1) if pl_total_samples > 0 else 0.0
            print(f"  │ Warmup: {warmup_batches_used} batches (ent+div only), "
                  f"then PL/KD for {n_batches - warmup_batches_used} batches")
            print(f"  │ PL: {pl_avg_rate:.1f}% agreed, "
                  f"{pl_acc_avg:.1f}% of PLs correct | "
                  f"Teacher err: {teacher_err_avg:.1f}% | "
                  f"voters={current_fdd_domain + 1}")
            print(f"  │ PL_loss={pl_loss_sum / max(pl_total_samples, 1):.4f} | "
                  f"KD_loss={kd_loss_sum / max(pl_total_samples, 1):.4f}")
        print(f"  │ Step: {global_step}/{total_batches}")
        if takeover_batch is not None:
            print(f"  │ ⚠ TAKEOVER: cls{top_class_history[takeover_batch][0]} at batch {takeover_batch}")
        else:
            print(f"  │ ✓ No single-class takeover")
        if trend > 10:
            print(f"  │ ⚠ DEGRADING within domain")
        print(f"  └{'─'*70}")

    # ─── Final summary ──
    print(f"\n{'='*90}")
    mean_error = np.mean([r["error"] for r in results.values()])
    if baseline_errors:
        print("  FINAL RESULTS: Backbone vs TTA")
        print(f"{'='*90}")
        print(f"\n  {'Domain':<25} {'Backbone':>10} {'TTA':>10} {'Improv.':>10} {'FDD':>5}")
        print(f"  {'─'*65}")
        for name, r in results.items():
            b = baseline_errors.get(name, 0); t = r["error"]; imp = b - t
            m = " ⚠" if imp < -1 else ""
            print(f"  {name:<25} {b:>9.1f}% {t:>9.1f}% "
                  f"{'↓' if imp > 0 else '↑'} {abs(imp):>7.1f}%{m:>3s} {r['fdd_domain']:>5d}")
        bm = np.mean(list(baseline_errors.values())); mi = bm - mean_error
        print(f"  {'─'*65}")
        print(f"  {'MEAN':<25} {bm:>9.1f}% {mean_error:>9.1f}% "
              f"{'↓' if mi > 0 else '↑'} {abs(mi):>7.1f}%")
    else:
        print("  FINAL RESULTS")
        print(f"{'='*90}")
        print(f"\n  {'Domain':<25} {'TTA Error':>12} {'FDD':>5}")
        print(f"  {'─'*45}")
        for name, r in results.items():
            print(f"  {name:<25} {r['error']:>11.1f}% {r['fdd_domain']:>5d}")
        print(f"  {'─'*45}")
        print(f"  {'MEAN':<25} {mean_error:>11.1f}%")

    print(f"\n  FDD domains: {fdd.num_domains}")
    if cfg.dataset in ["imagenet_plus", "imagenet_plusplus"]:
        _compute_rf(results, cfg)

    save_data = {"tta": dict(results), "mean_tta_error": mean_error,
                 "fdd_domains": fdd.num_domains}
    if baseline_errors:
        save_data["baseline"] = baseline_errors
        save_data["mean_backbone_error"] = bm; save_data["mean_improvement"] = mi
    os.makedirs(cfg.output_dir, exist_ok=True)
    p = os.path.join(cfg.output_dir, f"{cfg.dataset}_seed{cfg.seed}.json")
    with open(p, "w") as f: json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {p}")
    print(f"\n  FDD mapping:")
    for fid, names in domain_map.items():
        print(f"    FDD {fid} → {list(set(n.rsplit('_R',1)[0] for n in names))}")
    return results


def _compute_rf(results, cfg):
    from collections import defaultdict
    dr = defaultdict(list)
    for n, r in results.items(): dr[n.rsplit("_R", 1)[0]].append(r["error"])
    print(f"\n  RF analysis:")
    rv = []
    for b, e in dr.items():
        if len(e) >= 2:
            rf = e[-1] - e[0]; rv.append(rf)
            print(f"    {b:20s}: {e[0]:.1f}% → {e[-1]:.1f}% RF={rf:+.1f}")
    if rv: print(f"    Mean RF: {np.mean(rv):+.1f}")


if __name__ == "__main__":
    cfg = get_cfg(); set_seed(cfg.seed); adapt(cfg)
