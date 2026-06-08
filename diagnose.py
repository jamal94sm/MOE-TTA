"""
diagnose.py — Run this to identify why error is too high.
Tests 5 things in sequence, each isolating one potential failure.

Usage:
  python diagnose.py --data_dir ./data/ImageNet-C --corruptions gaussian_noise
"""

import torch
import torch.nn.functional as F
import timm
from tqdm import tqdm

from config import get_cfg
from datasets import get_domain_sequence
from model import build_model


def accuracy_on_loader(model_fn, loader, device, max_batches=20):
    """Run model_fn on a few batches and return error rate."""
    correct = 0; total = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model_fn(imgs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    err = 100.0 * (1 - correct / total)
    acc = 100.0 * correct / total
    return err, acc, total


def main():
    cfg = get_cfg()
    device = cfg.device
    domain_sequence = get_domain_sequence(cfg)
    domain_name, loader = domain_sequence[0]
    print(f"Diagnosing on: {domain_name} ({len(loader.dataset)} samples)\n")

    # ─── Test 1: Frozen pretrained ViT-Base with NO experts ───────────
    print("=" * 60)
    print("TEST 1: Pretrained ViT-Base — NO experts, NO adaptation")
    print("  (This is the 'Source' baseline from Table 1)")
    print("=" * 60)
    
    backbone = timm.create_model(cfg.backbone, pretrained=True, 
                                  num_classes=cfg.num_classes)
    backbone = backbone.to(device).eval()
    
    err, acc, n = accuracy_on_loader(backbone, loader, device)
    print(f"  Error: {err:.1f}%  Acc: {acc:.1f}%  (on {n} samples)")
    print(f"  Paper Source baseline for gaussian_noise: 53.0%")
    if err > 60:
        print("  ⚠ ERROR: Pretrained model itself is bad!")
        print("    → Check that timm is loading pretrained weights")
        print("    → Check image preprocessing (Resize 256, CenterCrop 224, ImageNet norm)")
    elif err < 56:
        print("  ✓ Pretrained model matches expected range")
    print()

    # ─── Test 2: Model with experts but BEFORE any adaptation ─────────
    print("=" * 60)
    print("TEST 2: Model with experts inserted — BEFORE adaptation")
    print("  (Expert bypass output should be ~0 at initialization)")
    print("=" * 60)
    
    model = build_model(cfg)
    model.eval()
    
    # Create one domain expert (as main.py does on first batch)
    model.expand_domain()
    model.set_active_domain(0)
    
    err, acc, n = accuracy_on_loader(model, loader, device)
    print(f"  Error: {err:.1f}%  Acc: {acc:.1f}%  (on {n} samples)")
    if err > 60:
        print("  ⚠ ERROR: Expert bypass degrades the model even at init!")
        print("    → The randomly initialized experts are injecting noise")
        print("    → Need to verify zero-init of expert up-projection")
    else:
        print("  ✓ Experts don't hurt at initialization")
    print()

    # ─── Test 3: Measure expert bypass magnitude vs MLP magnitude ─────
    print("=" * 60)
    print("TEST 3: Expert output magnitude vs backbone MLP output")
    print("  (Expert bypass should be << MLP output at initialization)")
    print("=" * 60)
    
    imgs, _ = next(iter(loader))
    imgs = imgs[:4].to(device)  # just 4 images
    
    # hook into one block to measure
    blocks = list(model.backbone.blocks)
    expert_mods = list(model.expert_modules)
    
    with torch.no_grad():
        # get features at the MLP input of block 0
        # run partial forward to get intermediate features
        x = model.backbone.patch_embed(imgs)
        if hasattr(model.backbone, 'cls_token'):
            cls = model.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = model.backbone.pos_drop(x + model.backbone.pos_embed)
        
        block = blocks[0]
        # attention + residual
        x_after_attn = x + block.attn(block.norm1(x))
        mlp_input = block.norm2(x_after_attn)
        
        # original MLP output
        mlp_out = block.mlp(mlp_input)
        
        # expert bypass output
        expert = expert_mods[0]
        expert_full = expert(mlp_input)      # = input + λ*shared + (1-λ)*domain
        expert_bypass = expert_full - mlp_input  # = λ*shared + (1-λ)*domain
        
        mlp_norm = mlp_out.norm(dim=-1).mean().item()
        expert_norm = expert_bypass.norm(dim=-1).mean().item()
        ratio = expert_norm / (mlp_norm + 1e-8)
        
        print(f"  MLP output norm:     {mlp_norm:.4f}")
        print(f"  Expert bypass norm:  {expert_norm:.4f}")
        print(f"  Ratio (expert/MLP):  {ratio:.4f}")
        if ratio > 0.1:
            print(f"  ⚠ WARNING: Expert bypass is {ratio:.1%} of MLP output!")
            print(f"    → This noise is added to EVERY block × EVERY token")
            print(f"    → With 12 blocks, cumulative noise destroys predictions")
            print(f"    → Fix: ensure up-projection is zero-initialized")
        else:
            print(f"  ✓ Expert bypass is small relative to MLP")
    print()

    # ─── Test 4: Check which parameters are trainable ─────────────────
    print("=" * 60)
    print("TEST 4: Trainable parameter check")
    print("=" * 60)
    
    trainable = model.get_trainable_params()
    frozen_backbone = sum(p.numel() for p in model.backbone.parameters() 
                          if not p.requires_grad)
    trainable_expert = sum(p.numel() for p in trainable)
    total_backbone = sum(p.numel() for p in model.backbone.parameters())
    
    print(f"  Backbone params (frozen): {frozen_backbone:,}")
    print(f"  Expert params (trainable): {trainable_expert:,}")
    print(f"  Total backbone params: {total_backbone:,}")
    
    # check no backbone params are trainable
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() 
                             if p.requires_grad)
    if backbone_trainable > 0:
        print(f"  ⚠ WARNING: {backbone_trainable:,} backbone params are trainable!")
        print(f"    → The backbone should be fully frozen")
    else:
        print(f"  ✓ Backbone is fully frozen")
    print()

    # ─── Test 5: Run a few adaptation steps and track loss + error ────
    print("=" * 60)
    print("TEST 5: Adaptation dynamics (20 batches)")
    print("  Tracking: loss, error, expert norm growth")
    print("=" * 60)
    
    model = build_model(cfg)
    model.expand_domain()
    model.set_active_domain(0)
    
    params = model.get_trainable_params()
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
    
    print(f"  {'Batch':>5} {'Loss':>8} {'Error%':>8} {'Expert_norm':>12} {'Entropy':>10}")
    print(f"  {'-'*48}")
    
    for i, (imgs, labels) in enumerate(loader):
        if i >= 20:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        
        # forward
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # filtered entropy loss
        mask = entropy < cfg.entropy_threshold
        if mask.sum() > 0:
            loss = entropy[mask].mean()
        else:
            loss = entropy.mean()
        
        # accuracy
        preds = logits.argmax(dim=-1)
        err_pct = 100.0 * (1 - (preds == labels).float().mean().item())
        
        # expert norm (check growth)
        expert_norms = []
        for em in model.expert_modules:
            for moe in [em.shared_moe] + list(em.domain_moes):
                for expert in moe.experts:
                    expert_norms.append(expert.up.weight.data.norm().item())
        avg_expert_norm = sum(expert_norms) / len(expert_norms)
        
        print(f"  {i:5d} {loss.item():8.4f} {err_pct:8.1f} {avg_expert_norm:12.6f} "
              f"{entropy.mean().item():10.4f}")
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print()
    print("=" * 60)
    print("DIAGNOSIS COMPLETE — check the ⚠ warnings above")
    print("=" * 60)


if __name__ == "__main__":
    main()
