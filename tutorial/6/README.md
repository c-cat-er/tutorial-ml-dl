## 6. Unsupervised Face Anomaly Detection System

- version: p6_1
- Ref: [prj-6-anomaly-detection](https://docs.google.com/document/d/13bdI8X4YXPaK46HETLY-L9XtddaqQ90PtWj6Ope5qMQ/edit?tab=t.apvy3tsjbwb3)

## 1. Project Overview

- Task Type: Unsupervised / One-Class Image Anomaly Detection.
- Core Concept: Anomaly scoring bounded by reconstruction error. The network is trained exclusively on normal human face images to master the low-dimensional compression and structural restoration of normal facial topology. When exposed to unseen anomalous distributions during inference, the decoder fails to accurately reconstruct the inputs, yielding elevated reconstruction errors designated as the "Anomaly Score."
- Evaluation Framework:
    - Inference Anomaly Metric: Root Mean Squared Error (RMSE) mapped over independent pixel dimensions.
    - Competition Evaluation Metric: Area Under the ROC Curve (ROC-AUC) assessed via the submission server.
- Project Constraints: Relying strictly on the provided closed dataset. Utilizing external training assets or importing pre-trained model weights is strictly prohibited.

---

##2. Data Architecture & Preprocessing

- Dataset Manifest (`data-bin/`):
    - Training Split (`trainingset.npy`): Unlabeled tensor containing exclusively normal human face image matrices.
    - Testing Split (`testingset.npy`): Mixed binary benchmarking split containing both normal and anomalous image variations.
- Custom Dataset Wrapper (`CustomTensorDataset`):
    - Channel Permutation: Automatically permutes raw image matrices from standard standard HWC arrays into PyTorch-compliant CHW tensors (Channels, Height, Width).
    - Dynamic Value Scaling: Maps raw integer pixel intensity bounds \([0, 255]\) into continuous floating-point vectors spanning \([-1.0, 1.0]\) to perfectly match the operational range of the decoder's final `Tanh` activation function.

---

## 3. Model Architectures & Ablation Topology

- The framework integrates four standalone autoencoder topologies to facilitate comprehensive architectural ablation studies:
    -   1. Fully-Connected Autoencoder (`fcn_autoencoder`): A baseline Multilayer Perceptron (MLP) pipeline that flattens 2D inputs into 1D arrays for bottlenecks and symmetric dimension expansion.
    -   2. Convolutional Autoencoder (`conv_autoencoder`): A fully convolutional pipeline leveraging paired Conv2d and ConvTranspose2d layers. Integrates structural `BatchNorm2d` layers across intermediate representations to regularize internal covariate shifts and stabilize deep gradient descent.
    -   3. Variational Autoencoder (`VAE`): Leverages the reparameterization trick to map spatial feature distributions onto structural continuous latent space parameters (mean μ and log variance \(\log\sigma^2\)). Optimized via a coupled target framework tracking MSE reconstruction losses and Kullback-Leibler Divergence (KLD).
    -   4. ResNet Autoencoder (`Resnet`): Deploys a fully initialized-from-scratch ResNet-18 skeleton (with the trailing dense classification head omitted) as a powerful feature mapping encoder. Features a mirrored decoder network utilizing transposed convolutions and structural bilinear interpolation for shape restoration.

---

## 4. Training & Optimization Setup

- Reproducibility Constraints: Enforces rigorous random initialization masking via `same*seeds(2000)` to freeze Python, NumPy, PyTorch CPU/GPU backends, and deterministic CuDNN benchmarking state flags for absolute experimental duplication.
- Hyperparameter Specifications:
    - Optimizer: Standard `Adam` configured with a constant learning rate lr = 1 × 10⁻⁴.
    - Loss Formulation: `nn.MSELoss()` for fcn, cnn, and resnet variants; a custom pixel-wise averaged MSE + KLD joint cost solver for the VAE pipeline.
    - Batch Capacity & Budget: Structured batch size of 128 running over an execution horizon of 100 epochs.
    - Checkpoint Tracking: Automatically tracks batch loss performance to dynamically persist historical optimal parameters (`best_model*{model*type}.pt`) alongside final epoch parameters (`last_model*{model_type}.pt`).

---

## 5. Inference & Anomaly Scoring Pipeline

- Memory Isolation: Freezes active parameter weights by executing `model.eval`(), coupled with `with torch.no_grad()` execution contexts to block backpropagation graphs and compress operational VRAM overhead.
- Pixel-Wise Error Processing: Utilizes `nn.MSELoss(reduction="none")` to calculate individual sample errors independently, flattening spatial errors via sequence summation across active pixel dimensions (`dim=[1, 2, 3]`).
- RMSE Transformation: Transforms sample-level pixel deviations through a element-wise square-root operation to output standardized one-dimensional Anomaly Scores. Higher output scores directly indicate a greater mathematical probability of distribution deviation.
- Output Logging: Flattens sample prediction scores and serializes the anomaly index table directly into a Kaggle-compliant format named `PREDICTION_FILE.csv`.
