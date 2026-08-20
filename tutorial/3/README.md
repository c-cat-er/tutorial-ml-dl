# 3. Food-11 Semi-Supervised Image Classification System

- version: p3_1
- Ref: [prj-3-image-classification](https://docs.google.com/document/d/1tS_9-Puw-fdOlz6C83FlwmDoSDp1OPW4rsILYc1N65M/edit?tab=t.3fadawov2tzq)

## 1. Project Overview

- Task Type: 11-class multiclass image classification.
- Core Challenge: Elevating model generalization and classification accuracy by effectively leveraging a substantial volume of unlabeled data when manually annotated training samples are constrained.
- Key Components: From-scratch ResNet18 backbone, semi-supervised pseudo-labeling (self-training), dual-view data augmentation pipeline, label smoothing regularization, and cosine annealing learning rate scheduling.

---

## 2. Data Architecture & Preprocessing

- Dataset Organization (`food-11`):
    - Labeled Training Directory: `food-11/training/labeled` (partitioned into 11 class folders).
    - Unlabeled Training Directory: `food-11/training/unlabeled` (used for semi-supervised expansion).
    - Validation Directory: `food-11/validation` (used for hyperparameter assessment).
    - Testing Directory: `food-11/testing` (used for final performance evaluation).
- Dual-View Augmentation Framework:
    - Strong Augmentation (`train_tfm`): Applied to labeled data and the verified pseudo-labeled subset during training to artificially expand sample diversity.
    - Resize((128, 128)) → RandomHorizontalFlip(p=0.5) → RandomRotation(15) → ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) → RandomResizedCrop(128, scale=(0.8, 1.0)) → ToTensor().
- Deterministic Transformation (test_tfm): Applied to validation, testing, and the pseudo-label inference stage to remove stochastic noise and provide stable evaluation baselines.
    - `Resize((128, 128)) → ToTensor()`.

---

## 3. Model Architecture

- Feature Extraction Backbone: Standard `torchvision.models.resnet18(weights=None)`. The model is initialized completely from scratch; the usage of any pre-trained weights is strictly prohibited.
- Fine-Tuning Classifier Head:
    - Dynamically extracts the input feature dimension (`in_features`) from the final fully-connected layer of the ResNet18 backbone.
    - Reconfigures the linear output projection layer as `nn.Linear(in_features, 11)` to match the 11 target food categories.

---

## 4. Semi-Supervised Pseudo-Labeling Mechanism

- Warm-up Phase: Restricts training exclusively to the raw labeled training dataset for the first 5 epochs, allowing the network to establish reliable baseline feature representations.
- Confidence Threshold Filtering: Enforces a strict selection threshold of τ = 0.85 to filter out low-confidence predictions.
- Dual-View Selection Pipeline:
    -   1. Label Generation (Clean View): Prior to each epoch (from epoch 6 onwards), the model switches to `model.eval()`. It processes the unannotated images passed through the deterministic `test_tfm` (`unlabeled_set_pseudo`). The target pseudo-labels and their indices are retained only if the maximum Softmax probability exceeds 0.85.
    -   2. Augmented Training (Strong View): Utilizing the extracted high-confidence indices, the system pulls the corresponding images from the strongly augmented `unlabeled_set` (`train_tfm`). These are wrapped into a custom PseudoDataset and dynamically merged with the original labeled dataset via ConcatDataset to reconstruct the active `train_loader`.

---

## 5. Training & Optimization Strategy

- Optimizer Configuration:
    - Optimizer: AdamW with an initial learning rate lr = 10⁻³ and weight decay coefficient of 10⁻⁵ to rectify mathematical weight decay discrepancies found in traditional Adam solvers.
    - Regularization: `CrossEntropyLoss` adjusted with `label_smoothing=0.1` to redistribute 10% of uniform probability across incorrect classes, mitigating overconfidence and suppressing overfitting.
    - Scheduler: `CosineAnnealingLR` with \(T\_{\max} = 80\) epochs for smooth learning rate convergence.
- Hardware & I/O Optimization:
    - Batch Size: 128.
    - Total Run Budget: 80 epochs.
    - Execution Pipeline: `DataLoader` instances are deployed with `num_workers=2` and `pin_memory=True` to activate Direct Memory Access (DMA), accelerating CPU-to-GPU tensor transfers.

---

## 6. Inference & Evaluation Pipeline

- Environment Freezing: Switches the model state to `model.eval()` to lock down - BatchNorm statistics and disable Dropout layers during testing.
- Accelerated Forward Pass: Implements dummy target labels (defaulted to 0) to comply with `DatasetFolder` structural formatting constraints on unlabeled testing folders. Wraps the batched loop inside a with `torch.no_grad()` context block to eliminate gradient graph tracking, maximize inference speed, and compress VRAM footprint.
- Submission Generation: Aggregates batch-level predictions via `logits.argmax(dim=-1)` and exports the final sequence to a submission-ready format: `predict.csv`.
