# 7. Unsupervised Domain Adversarial Training System (DaNN)

- version: p7_1
- Ref: [prj-7-DaNN](https://docs.google.com/document/d/1CdxoApT-yJ6I5Ji9gZ1SFgw5Ft5DXMnZfNyVOymwySI/edit?tab=t.6aqjo8ecx6up)

## 1. Project Overview

- Task Type: Unsupervised Domain Adaptation (UDA) / Domain Adversarial Transfer Learning.
- Core Challenge: Domain Shift & Gap. The distribution of color, texture, and background varies significantly between the source and target domains. The network must learn domain-invariant category features without relying on target-domain labels.
- Dataset Manifest:
    - Source Domain (Real Photos): 5,000 labeled RGB images (\(32 \times 32\)) distributed uniformly across 10 balanced categories (0–9).
    - Target Domain (Drawings): 100,000 unlabeled grayscale images (\(28 \times 28\)) mapping onto the same 10 balanced categories.
- Evaluation Framework:
    - Primary Metric: Target Domain Classification Accuracy.
    - Core Prior Knowledge: The class distributions in both the source training data and target testing data are perfectly uniform (\(1:1:1...\)). This perfect balance allows for post-processing Prior Shift Correction.
- Project Constraints: Relying strictly on the closed competition dataset. Utilizing external training data or importing pre-trained model weights on external datasets is strictly prohibited.

---

## 2. Data Architecture & Preprocessing Pipeline

- Domain Alignment & Edge Detection (Canny Filter): Leverages OpenCV's `cv2.Canny` algorithm on the source domain to strip complex backgrounds and colors. Forcing the source images into edge-line structures drastically bridges the feature mismatch between real photos and drawings.
    - Channel & Dimension Normalization: Converts images to single-channel tensors using `transforms.Grayscale()` (since Canny filters accept single-channel inputs).
    - Enlarges target drawing matrices from \(28 \times 28\) to \(32 \times 32\) via `transforms.Resize()` to perfectly align feature tensor dimensions.
- Data Augmentation: Applies a stochastic training pipeline containing a 50% `RandomHorizontalFlip` and a \(\pm 15^{\circ }\) `RandomRotation` (padding empty rotated border pixels with absolute black via `fill=(0,)`).
- Parallel Dual-Stream Data Loading: Employs `ImageFolder` to dynamically map directory structures into task labels. Configures parallel `source*dataloader` and `target_dataloader` setups with synchronized batch capacities (\(\text{batch_size}=32\)) and `shuffle=True`.

---

## 3. Model Architecture & Tri-Head Topology

- The network features a Tri-Head Domain Adversarial Neural Network (DaNN) configuration:
    - Feature Extractor (`FeatureExtractor`): A 5-layer Convolutional Neural Network (CNN) interleaved with `nn.BatchNorm2d` and `nn.MaxPool2d` layers to encode inputs into a 512-dimensional latent feature vector.
    - Defensive Squeeze Optimization: The forward pass utilizes an explicit dimension collapse `squeeze(-1)` instead of a generic `squeeze()`. This strictly prevents shape-mismatch crashes during single-sample inference (when \(\text{Batch}=1\)) by preserving the batch dimension.
- Label Predictor (`LabelPredictor`): A classic Multilayer Perceptron (MLP) classification head (\(512 \to 512 \to 512 \to 10\)) that maps latent representations into 10 class logits.
- Domain Classifier (`DomainClassifier`): A deep binary classification network heavily regularized with dense nn.BatchNorm1d segments. It calculates the probability of whether an encoded latent feature originates from the source domain (Label 1) or target domain (Label 0).

---

## 4. Training & Optimization Setup

- Reproducibility Constraints: Fixes experimental variance using `SEED = 42` across Python random, NumPy, PyTorch CPU/GPU backends, and CuDNN execution flags to establish 100% computational duplication.
- Two-Stage Decoupled Adversarial Optimization:
    - BatchNorm Anti-Pollution (Data Mixing): Since the running statistics (mean/variance) of the two domains vary wildly, processing them sequentially destabilizes BatchNorm tracking parameters. The pipeline horizontally concatenates the source and target mini-batches (`torch.cat`) into a single `mixed_data` stream before feeding it into the network.
    - Step 1 (Domain Discriminator Updates): Extracts mixed latent vectors and applies .detach() to isolate the feature extractor graph. Updates the DomainClassifier to maximize domain recognition accuracy using `nn.BCEWithLogitsLoss`.
    - Step 2 (Adversarial Feature Alignments): Re-extracts features with active gradients. Optimized via an adversarial cost objective function: \(\mathcal{L}\*{F} = \mathcal{L}_{\text{classification}} - \lambda \cdot \mathcal{L}_{\text{domain}}\). The negative sign acting on the domain loss mimics a Generator in a Generative Adversarial Network (GAN), punishing the Feature Extractor if the Domain Classifier succeeds, thus forging Domain-Invariant Features.
- Hyperparameters & Scheduling:
    - Optimizers: Independent `Adam` optimizers configured with an initial \(lr = 10^{-4}\) and a balancing weight coefficient \(`\lambda = 0.1`\).
    - LR Schedulers: Integrated `lr*scheduler.StepLR` engines that cut the learning rate by half (`gamma=0.5`) every 50 epochs to guarantee smooth late-stage model convergence across 200 epochs.

---

## 5. Evaluation & Post-Processing Calibration (Strong Baseline)

- Stratified Validation & Early Stopping: Employs `train_test_split` regularized with class stratification matching `source_dataset.targets` to extract a clean 10% source validation set. Implements an early stopping barrier set to a patience window of 20 un-improved validation epochs.
- Prior Shift Post-Processing Calibration:
    - Concentrates batch-level evaluation matrices over the 100,000 target samples under an isolated `model.eval()` and `torch.no_grad()` pipeline, assembling a comprehensive target logit tensor.
    - Computes the mean soft probability distribution (\(pred_dist\)) across all target inferences.
    - Applies Prior Shift Correction by adjusting raw logits: \(\text{logits}\*{\text{calibrated}} = \text{logits} - \log(pred_dist + \epsilon)\). This mathematically penalizes classes that are over-predicted due to model bias and compensates under-predicted classes, steering the global testing distribution toward a perfect uniform distribution (\([0.1, 0.1, ..., 0.1]\)).
    - Implements both single-pass correction (`calibrate_by_distribution`) and recursive adjustments (`calibrate_iterative`) to guarantee maximum calibration stability.
- Output Tracking: Maps calibrated predictions into sequential test indexes and serializes table sequences into `DaNN_submission_strong_baseline.csv`.
