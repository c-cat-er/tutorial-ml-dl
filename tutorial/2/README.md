# 2. TIMIT Framewise Phoneme Classification System

- version: p2_1
- Ref: [prj-2-timi-framewise-phoneme-classification](https://docs.google.com/document/d/1I68MpI8M8ims4KFPOOrf7LirViSfuon0Mj7F-T6lMMg/edit?tab=t.0)

## 1. Project Overview

- Task Type: Framewise audio sequence multiclass classification (39 distinct phoneme classes).
- Objective: Predict the corresponding phoneme class for each individual speech frame using Mel-Frequency Cepstral Coefficients (MFCC) extracted from the TIMIT corpus.
- Evaluation Metric: Frame-level Classification Accuracy.
- Key Components: Context Window Feature Concatenation, 5-Fold Cross-Validation, AdamW Optimizer, Label Smoothing, and Multi-Fold Logits Averaging Ensemble.

---

## 2. Data Architecture & Preprocessing

- Dataset Components:
    - Training data: `train_11.npy` and target labels: `train_label_11.npy`.
    - Unlabeled testing data: `test_11.npy`.
- Context Window Configuration:
    - Each temporal frame comprises a 39-dimensional MFCC feature vector.
    - Symmetrically concatenates 5 preceding and 5 succeeding frames around the central frame (11 frames in total), resulting in a finalized input feature dimension of \(11 \times 39 = 429\).
- Standardization Pipeline:
    - Computes the mean (\(\mu \)) and standard deviation (\(\sigma \)) dynamically for each fold's training split to perform Z-score normalization.
    - Exports normalization statistics per fold (`foldX_norm.npz`) to guarantee identical feature scaling during the inference stage.

---

## 3. Model Architecture

- Network Type: 5-layer Multilayer Perceptron (MLP) Classifier.
- Structural Configuration:
    - Input Layer: 429 dimensions.
    - Hidden Layers: Sequential topology of \(429 \to 1024 \to 512 \to 256 \to 128\) mapped via `GELU` activation functions.
    - Regularization: Integrates BatchNorm1d and progressive Dropout rates (\(0.4 \to 0.3 \to 0.3 \to 0.2\)) across hidden segments to prevent overfitting.
    - Output Layer: Dense mapping from \(128 \to 39\) dimensions (corresponding to the 39 phoneme categories).

---

## 4. Training & Optimization Strategy

- Optimizer & Loss Setup:
    - Optimizer: `AdamW` with an initial learning rate \(lr = 5 \times 10^{-4}\) and weight decay coefficient of \(1 \times 10^{-4}\).
    - Loss Function: `CrossEntropyLoss` combined with a Label Smoothing factor of \(0.1\) to avoid overconfident predictions.
    - Learning Rate Scheduler: `ReduceLROnPlateau` monitoring validation loss (decay factor = \(0.5\), patience = \(6\) epochs).
- Validation & Overfitting Controls:
    - Robust Evaluation: Bound by a standard 5-Fold Cross-Validation scheme.
    - Early Stopping: Triggers early termination if validation accuracy fails to improve for 20 consecutive epochs.
    - Memory Management: Utilizes PyTorch `DataLoader` (`num_workers=4, pin_memory=True`) and invokes explicit garbage collection (`gc.collect()`) after completing each fold to clear GPU/RAM overhead.

---

## 5. Inference & Ensemble Pipeline

- Model Restoration: Sequentially loads the optimal state checkpoints saved from all training instances (`fold1_best.ckpt` to `fold5_best.ckpt`).
- Normalized Inference: Normalizes the raw testing samples frame-by-frame using the corresponding fold-specific statistics (\(\mu \) and \(\sigma \)) prior to forward passes.
- Logits Average Ensemble: Accumulates the unnormalized output logs (logits) from all 5 fold models, computes the arithmetic mean (`avg_logits`), and extracts final predictions via the `argmax` operation.
- Output Export: Formats and flattens prediction outputs into a submission-ready CSV file named `prediction5.csv`.
