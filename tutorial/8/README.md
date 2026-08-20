# 8. Lightweight Food Image Classification via Knowledge Distillation and Efficient Architecture Design

- version: p8_1
- Ref: [prj-8-network-compression](https://docs.google.com/document/d/1KXtY1kLxddd_nQVyH_dSZV79MFFdux-KqtiYys6f3G4/edit?tab=t.gii6wy7eh2jb)

## 1. Project Overview

- Task Type: Lightweight Multiclass Image Classification (11 distinct categories).
- Core Challenge: Extreme Resource Constraints. The network must achieve high classification accuracy while strictly adhering to a maximum parameter capacity of ≤ 100,000 total parameters (including both trainable and non-trainable weights).
- Evaluation Framework:
    - Evaluation Metric: Top-1 Classification Accuracy evaluated via the submission server.
    - Strict Regulatory Constraints: Leveraging external datasets, downloading online labels, or importing standard weights pre-trained on external computer vision benchmarks is strictly prohibited. Test data can only be utilized for inference; generating pseudo-labels on the test split to overfit student models is an honor code violation.
- Key Methodology: Deploying an optimized Student-Teacher Knowledge Distillation (KD) framework and restructuring standard convolutional blocks into highly efficient Depthwise Separable Convolutions (DSC).

---

## 2. Data Architecture & Preprocessing Pipeline

- Dataset Structure (`food-11`):
    - Labeled Training Split: 280 images × 11 categories = 3,080 annotated images.
    - Unlabeled Expansion Split: 6,786 unannotated images reserved for optional semi-supervised self-training pipelines.
    - Validation Split: 60 images × 11 categories = 660 validated images.
    - Testing Split: 3,347 benchmark images.
- Data Augmentation Strategies (`Dual-View Transforms`):
    - Stochastic Training Pipeline (`train\*tfm`): Prevents overfitting by adding spatial and color perturbation: `Resize((142, 142)) → RandomHorizontalFlip(p=0.5) → RandomRotation(20) → ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) → RandomCrop(128) → ToTensor()`.
    - Deterministic Transformation (`test_tfm`): Stabilizes late-stage evaluation and inference: `Resize((142, 142)) → CenterCrop(128) → ToTensor()`.
- I/O Pipeline Optimization: Configures `DataLoader` pipelines using batch_size=64 coupled with `pin_memory=True` to activate direct hardware DMA tensor streaming onto memory units.

---

### 3. Efficient Model Architecture (`StudentNet`)

- To strictly respect the 100k parameter barrier while retaining high representational capacity, the standard spatial convolution layers are completely replaced with a custom MobileNet-style block layout:
    - Depthwise Separable Convolution (`dwpw_conv`): Mapped as a decoupled two-stage kernel tracking pipeline:
        -   1. Depthwise Phase: A standard `nn.Conv2d` layer matching kernel size 3 × 3 with channel grouping activated (`groups=in_chs`) to isolate intra-channel spatial patterns.
        -   2. Pointwise Phase: A linear 1 × 1 `nn.Conv2d` layer that projects aggregated channel structures into target dimension manifolds.
        - Both phases are tightly bounded by `nn.BatchNorm2d` and `nn.ReLU(inplace=True)` sequences to maximize normalization stability.
- Global Network Topology:
    - Input Feature Map: 3 × 128 × 128.
    - Feature Core: Standard stem `Conv2d (3 → 32) → dwpw_conv` sequences progressively expanding and downsampling feature resolutions (32 → 64 → 128 → 256) down to a final spatial footprint of 256 × 8 × 8.
    - Pooling & Output Classification: Collapses spatial domains via `nn.AdaptiveAvgPool2d((1, 1)`) and flattens representations into a dense output classification layer mapping `nn.Linear(256, 11)`.

---

## 4. Knowledge Distillation & Training Framework

- Teacher Model Configuration: Restores an pre-trained high-capacity ResNet backbone framework (`teacher_net.ckpt`, achieving an accuracy baseline of ~0.788) frozen strictly under `model.eval()` to generate stable soft-target class probability guides.
- Formulated Loss Function (`loss_fn_kd`): Couples conventional classification errors with dark-knowledge transfer properties through an adjustable dual cost layout:

\(\mathcal{L}_{\text{KD}}=(1-\alpha )\cdot \mathcal{L}*{\text{CE}}(\^{y},y)+\alpha \cdot T^{2}\cdot \mathcal{L}*{\text{KLD}}\left(\log \sigma \left(\frac{z_{s}}{T}\right),\sigma \left(\frac{z\*{t}}{T}\right)\right)\)

- Hyperparameters: Temperature hyperparameter configured to T=6 to smoothly soften output logits, alongside a soft-target weighting multiplier set to α=0.7.
- Optimization Specifications:
    - Optimizer: `AdamW` deployed with a tuned base learning rate lr = 3 × 10⁻⁴ and weight decay coefficient 1 × 10⁻⁴ to enforce weight regularization.
    - Scheduling: Governed by a `CosineAnnealingLR` engine spanning \(T\*{\max} = 80\) epochs to facilitate smooth, late-stage gradient convergence.
    - Gradient Stabilization: Enforces a strict norm gradient ceiling via `nn.utils.clip_grad_norm\*(..., max_norm=10)` to neutralize exploding gradient trajectories during backpropagation.

---

## 5. Inference & Output Tracking Pipeline

- Environment Freezing: Commands the optimized `StudentNet` into an isolated `model.eval()` structure to freeze moving averages inside BatchNorm vectors.
- Testing Execution: Iterates testing tensors wrapped in a `with torch.no_grad()` execution context to suppress backpropagation graphs and maximize runtime throughput.
- Submission Logging: Extracts predicted index tables via `logits.argmax(dim=-1)` and serializes prediction entries directly into a Kaggle-compliant format named `predict.csv`.
