# Reproduction: Shared & Domain Self-Adaptive Experts with FDD for CTTA (AAAI 2026)

## 1. Environment Setup

```bash
conda create -n ctta python=3.9 -y && conda activate ctta
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.9.16 numpy scipy scikit-learn tqdm pillow
```

## 2. Dataset Preparation

### 2.1 ImageNet-C (15 corruptions × 5 severity levels, ~19 GB)
```bash
mkdir -p ./data/ImageNet-C && cd ./data/ImageNet-C
wget https://zenodo.org/records/2235448/files/blur.tar
wget https://zenodo.org/records/2235448/files/digital.tar
wget https://zenodo.org/records/2235448/files/noise.tar
wget https://zenodo.org/records/2235448/files/weather.tar
wget https://zenodo.org/records/2235448/files/extra.tar
for f in *.tar; do tar xf "$f"; done
```
Structure: `ImageNet-C/{corruption_name}/5/{class_folders}/`

### 2.2 CIFAR-100-C
```bash
mkdir -p ./data/CIFAR-100-C && cd ./data/CIFAR-100-C
wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar.gz
tar xzf CIFAR-100-C.tar.gz
```

### 2.3 CRS Benchmark (ImageNet+/++)
```bash
# ImageNet-V2 (Matched Frequency)
pip install imagenet-v2
# or: wget from https://huggingface.co/datasets/vaishaal/ImageNetV2

# ImageNet-A
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar xf imagenet-a.tar -C ./data/

# ImageNet-R
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar xf imagenet-r.tar -C ./data/

# ImageNet-Sketch
# Download from: https://github.com/HaohanWang/ImageNet-Sketch
```

### 2.4 ACDC (Segmentation)
Register at https://acdc.vision.ee.ethz.ch/ and download.

## 3. Running

```bash
# ImageNet-C standard CTTA
python main.py --dataset imagenet_c --data_dir ./data/ImageNet-C --severity 5

# ImageNet+ CRS benchmark  
python main.py --dataset imagenet_plus --data_dir ./data --num_rounds 3

# CIFAR-100-C
python main.py --dataset cifar100_c --data_dir ./data/CIFAR-100-C --severity 5
```

## 4. Key Hyperparameters (Appendix F)
| Parameter | Value |
|---|---|
| Backbone | ViT-Base/16 (timm, pretrained) |
| Shared expert rank | 32 |
| Domain expert rank | 16 |
| Experts per MoE module | M=2 |
| Fusion λ | 0.5 |
| FDD freq radius l | 16 |
| FDD threshold τ | 1.5 |
| Confidence κ | 0.4 × ln(C) |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| LR (ImageNet-C) | 1e-5 |
| LR (ImageNet+/++) | 5e-4 |
| Batch size | 50 |
| Weight decay | 0.05 |
| Covariance | Diagonal approximation |
