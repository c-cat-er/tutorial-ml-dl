# 1. COVID-19 Cases Prediction System

- version: p1_2
- Ref: [prj-1-covid19-prediction](https://docs.google.com/document/d/1J7HSalTLtpk0h2iekRgyNDHOODnHpvVo1-87OH6o21E/edit?tab=t.0)

## 1. Project Overview

- Objective: Predict the percentage of daily new positive cases on the third day based on historical data from US states (Regression task).
- Evaluation Metric: Mean Squared Error (MSE).
- Frameworks: PyTorch, NumPy, Scikit-learn, Optuna, Matplotlib.

---

## 2. Data Pipeline & Preprocessing

- Dataset: `covid.train.csv and covid.test.shuffle.csv` (93 raw features).
- Feature Selection (Strong Baseline): Extracted 42 features in total: 40 localized region indicators and 2 critical infection trackers (`tested_positive`).
- Data Splitting: Split training data into Train and Dev sets (9:1 ratio) using a deterministic random split (`seed=42069`).
- Preprocessing: Applied Z-Score Standardization (`mean=0, std=1`) exclusively to quantitative features (omitting the first 40 region columns).
- Data Loading: Wrapped in DataLoader with `batch_size=256`, automatic shuffling for training, and `pin_memory=True` for accelerated GPU memory access.

---

## 3. Model Architecture

- Network Type: Fully-Connected Deep Neural Network (DNN).
- Layer Sequence: Input (42) → Linear (64) → BatchNorm1d → ReLU → Dropout (0.3) → Linear (32) → BatchNorm1d → ReLU → Dropout (0.3) → Linear (16) → BatchNorm1d → ReLU → Dropout (0.2) → Linear Output (1).
- Regularization: Integrated Dropout layers and L2 Weight Decay to prevent overfitting.
- Loss Function: `nn.MSELoss(reduction="mean")`.

---

## 4. Training & Optimization Strategy

- Baseline Optimizer: Adam (`lr=0.0005, weight_decay=1e-5`).
- Early Stopping: Terminated training if validation loss did not improve for 80 consecutive epochs; automatically saved the optimal weights to `best_model.pth`.
- Automated Hyperparameter Tuning (Optuna Route):
    - Conducted a 30-trial optimization study to search for optimal learning rate, weight decay, dropout rate, and hidden dimensions.
    - Employed `MedianPruner` to dynamically terminate poor-performing trials early.
- Ensemble Strategy: Trained 5-Fold Cross-Validation models on the top-3 parameter sets, and applied Average Blending across estimators to generate final predictions.

---

## 5. Execution Pipeline & Outputs

- Data Download: Pull competition datasets locally via Google Drive setup.
- Model Training:
    -   1. Route A (Optuna Enabled): Runs the automated hyperparameter search and K-Fold ensemble pipeline.
    -   2. Route B (Optuna Disabled): Trains a standalone single model and visualizes performance via Learning Curves and Ground Truth vs. Prediction scatter plots.
- Inference & Submission: Performs test-set inference and outputs results to `pred.csv` (single model) or `pred_ensemble.csv` (K-Fold ensemble).

<style>
   my-red    { color: #d32f2f; font-weight: bold; }
   my-orange { color: #ed6c02; font-weight: bold; }
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; }
   my-green  { color: #2e7d32; font-weight: bold; }
   my-blue   { color: #0288d1; font-weight: bold; }
   my-cyan   { color: #00a8cc; font-weight: bold; }
   my-gray   { color: #8c8c8c; font-size: 0.9em; }
</style>
