Transfer Learning 中的 Domain Adversarial Training。
資料前處理 → Canny + Transform + ImageFolder + DataLoader。

## 1. 下載與解壓

- 利用 OpenCV 的 cv2.Canny 將來源<my-orange>圖片（Source）轉換成邊緣線條圖</my-orange>。
    - 對齊特徵分佈（Domain Alignment）：因為 Source（真實照片）與 Target（手繪圖/線條圖）的顏色、紋理分佈差異巨大（Domain Shift）。
    - 保留核心幾何語義：Canny 邊緣能強行濾除真實照片的顏色與複雜背景，僅留下形狀與輪廓，這能與手繪圖的特徵空間更接近，大幅降低 Domain Gap，屬於關鍵的特徵工程。

```bash
!gdown --id '1e4CaQ5VUF3F04XRDGXrnRQGogo89TiF8' --output real_or_drawing.zip
!unzip real_or_drawing.zip
```

## 2. Visualize the data 數據視覺化

```bash
import matplotlib.pyplot as plt

def no_axis_show(img, title="", cmap=None):
    # imshow, and set the interpolation mode to be "nearest"。
    fig = plt.imshow(img, interpolation="nearest", cmap=cmap)
    # do not show the axes in the images.
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)

titles = [
    "horse",
    "bed",
    "clock",
    "apple",
    "cat",
    "plane",
    "television",
    "dog",
    "dolphin",
    "spider",
]
plt.figure(figsize=(18, 18))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    fig = no_axis_show(
        plt.imread(f"real_or_drawing/train_data/{i}/{500 * i}.bmp"), title=titles[i]
    )
```

```bash
plt.figure(figsize=(18, 18))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    fig = no_axis_show(
        plt.imread(f"real_or_drawing/test_data/0/" + str(i).rjust(5, "0") + ".bmp")
    )
```

## 3. Special Domain Knowledge 領域知識 (Canny 邊緣檢測)

```bash
import cv2
import matplotlib.pyplot as plt

titles = [
    "horse",
    "bed",
    "clock",
    "apple",
    "cat",
    "plane",
    "television",
    "dog",
    "dolphin",
    "spider",
]
original_img = plt.imread("real_or_drawing/train_data/0/0.bmp")
gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(18, 18))

# original
plt.subplot(1, 5, 1)
no_axis_show(original_img, title="original")

# gray
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title="gray scale", cmap="gray")

# Canny with constants
CANNY_LOW, CANNY_HIGH = 170, 300
canny_img = cv2.Canny(gray_img, CANNY_LOW, CANNY_HIGH)
plt.subplot(1, 5, 3)
no_axis_show(canny_img, title=f"Canny({CANNY_LOW}, {CANNY_HIGH})", cmap="gray")

# 其他 threshold 示範
for idx, (low, high) in enumerate([(50, 100), (150, 200), (250, 300)], start=3):
    canny = cv2.Canny(gray_img, low, high)
    plt.subplot(1, 5, idx + 1)
    no_axis_show(canny, title=f"Canny({low}, {high})", cmap="gray")
plt.show()
```

## 4. Data Process

- Grayscale()：<my-orange>Canny 只支援單通道灰階圖，必須先轉灰階</my-orange>。
- Resize((32, 32))：Target 數據原本是 28x28，必須放大至 32x32，以確保與 Source 的特徵維度完全一致。
- RandomHorizontalFlip / RandomRotation：標準的數據增強，提升模型的泛化能力與魯棒性。

```bash
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

source_transform = transforms.Compose(
    [
        # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
        transforms.Grayscale(),
        # cv2 do not support skimage.Image, so we transform it to np.array,
        # and then adopt cv2.Canny algorithm.
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        # Transform np.array back to the skimage.Image.
        transforms.ToPILImage(),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ]
)
target_transform = transforms.Compose(
    [
        # Turn RGB to grayscale.
        transforms.Grayscale(),
        # Resize: size of source data is 32x32, thus we need to
        #  enlarge the size of target data from 28x28 to 32x32。
        transforms.Resize((32, 32)),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0,)),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ]
)

source_dataset = ImageFolder("real_or_drawing/train_data", transform=source_transform)
target_dataset = ImageFolder("real_or_drawing/test_data", transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
```

## 5. Model

- FeatureExtractor（特徵提取器）：負責從<my-orange>影像（灰階線條）中提取高階的潛在特徵（Latent Features）</my-orange>，輸出為 512 維向量。
- LabelPredictor（類別預測器）：標準的<my-orange>分類頭（Classification Head），根據提取的特徵來預測主任務的 10 個類別</my-orange>。
- DomainClassifier（領域分類器）：<my-orange>一個二分類頭，用來判別當前的特徵是來自 Source 還是 Target</my-orange>。

```bash
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y
```

## 6. Pre-processing 預處理與超參數設定

- StepLR(..., gamma=0.5)：<my-orange>動態調整學習率，每 50 個 Epoch 將學習率減半，幫助模型在後期穩定收斂</my-orange>。

```bash
import os
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LR = 1e-4
EPOCHS = 200
LAMBDA = 0.1
BATCH_SIZE = 32
CHECKPOINT_DIR = "checkpoints"

feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)
domain_classifier = DomainClassifier().to(device)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters(), lr=LR)
optimizer_C = optim.Adam(label_predictor.parameters(), lr=LR)
optimizer_D = optim.Adam(domain_classifier.parameters(), lr=LR)

# 新增 Scheduler
scheduler_F = optim.lr_scheduler.StepLR(optimizer_F, step_size=50, gamma=0.5)
scheduler_C = optim.lr_scheduler.StepLR(optimizer_C, step_size=50, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

## 7. Start Training

- 對抗訓練邏輯，代碼雖然沒有寫明 GRL（梯度反轉層）類別，但用<my-orange>兩階段分離最佳化（Two-stage Optimization）</my-orange>實現了相同效果
- Step 1：<my-orange>訓練 Domain Classifier</my-orange>
    - 特徵調用 .detach()：鎖定特徵提取器，只更新 Domain Classifier 的參數。
    - 目的：讓 Domain Classifier 盡可能精準地分出 Source（Label 1）和 Target（Label 0）。
- Step 2：<my-orange>訓練 Feature Extractor & Label Classifier</my-orange>
    - Loss 函數：loss_F = class_criterion(...) - lamb \* domain_criterion(...)。
    - 為什麼是「減號 -」？：主任務要最小化分類錯誤，但對 Domain Loss 而言，特徵提取器想要最大化 Domain Classifier 的錯誤（讓它混淆、分不出來）。這個減號就是對抗（Adversarial）的體現，<my-orange>迫使 FeatureExtractor 提取出領域無關（Domain-Invariant）且具備分類鑑別力的特徵</my-orange>。

```bash
def train_epoch(source_dataloader, target_dataloader, lamb):
    """
    Args:
      source_dataloader: source data的dataloader
      target_dataloader: target data的dataloader
      lamb: control the balance of domain adaptatoin and classification.
    """

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss = running_F_loss = total_hit = total_num = 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(
        tqdm(
            zip(source_dataloader, target_dataloader),
            total=min(len(source_dataloader), len(target_dataloader)),
        )
    ):
        source_data, source_label = source_data.to(device), source_label.to(device)
        target_data = target_data.to(device)

        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros(mixed_data.size(0), 1).to(device)
        domain_label[: source_data.size(0)] = 1

        # Step 1 : train domain classifier
        optimizer_D.zero_grad()
        feature = feature_extractor(mixed_data).detach()
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = domain_classifier(feature)
        loss_D = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss_D.item()
        loss_D.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        feature = feature_extractor(mixed_data)
        class_logits = label_predictor(feature[: source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss_F = class_criterion(class_logits, source_label) - lamb * domain_criterion(
            domain_logits, domain_label
        )
        loss_F.backward()
        optimizer_F.step()
        optimizer_C.step()
        running_F_loss += loss_F.item()

        total_hit += (torch.argmax(class_logits, dim=1) == source_label).sum().item()
        total_num += source_data.size(0)

    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num
```

## 8. Validation 驗證與早停機制

- <my-orange>使用 train_test_split 切出 10% 的 Validation 集合，並開啟 stratify（分層抽樣），保證驗證集的類別分佈與訓練集完全一致</my-orange>。
- patience = 20 的 Early Stopping：<my-orange>當驗證集準確率（val_acc）連續 20 個 Epoch 沒有改善時自動停止訓練</my-orange>，防止模型過擬合（Overfitting）。

```bash
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# === 新增：切出 validation set ===
val_ratio = 0.1
source_indices = list(range(len(source_dataset)))
train_idx, val_idx = train_test_split(
    source_indices,
    test_size=val_ratio,
    random_state=42,
    stratify=source_dataset.targets,
)

source_train_dataset = Subset(source_dataset, train_idx)
source_val_dataset = Subset(source_dataset, val_idx)

source_train_loader = DataLoader(
    source_train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False)


def validate(feature_extractor, label_predictor, val_loader, device):
    feature_extractor.eval()
    label_predictor.eval()
    total_hit, total_num = 0, 0

    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            features = feature_extractor(data)
            logits = label_predictor(features)
            total_hit += (torch.argmax(logits, dim=1) == label).sum().item()
            total_num += label.size(0)

    return total_hit / total_num


# === 新增：驗證後恢復訓練模式的輔助函數（可選）===
def set_train_mode(models):
    for model in models:
        model.train()
```

## 9. 主訓練迴圈

```bash
best_train_acc = 0.0
best_val_acc = 0.0
patience = 20  # 早停耐心值
no_improve_epochs = 0
early_stop = False

for epoch in range(EPOCHS):
    if early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

    # === 訓練階段 ===
    feature_extractor.train()
    label_predictor.train()
    domain_classifier.train()

    train_D_loss, train_F_loss, train_acc = train_epoch(
        source_train_loader, target_dataloader, LAMBDA
    )

    # === 驗證階段 ===
    feature_extractor.eval()
    label_predictor.eval()
    domain_classifier.eval()

    val_acc = validate(feature_extractor, label_predictor, source_val_loader, device)

    scheduler_F.step()
    scheduler_C.step()
    scheduler_D.step()

    # Checkpoint - 修正後版本
    if train_acc > best_train_acc:
        best_train_acc = train_acc          # ← 已修正 typo
        torch.save(
            {
                "epoch": epoch,
                "feature_extractor": feature_extractor.state_dict(),
                "label_predictor": label_predictor.state_dict(),
                "optimizer_F": optimizer_F.state_dict(),
                "optimizer_C": optimizer_C.state_dict(),
                "best_train_acc": train_acc,
            },
            f"{CHECKPOINT_DIR}/best_train_acc.pt"
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # 重置計數器，Early stopping 只看 validation metric（這裡是 val_acc）。
        # 只要 val_acc 有改善就重置 no_improve_epochs = 0。
        # 只有 val_acc 沒有改善時才累積 patience。
        no_improve_epochs = 0
        torch.save(
            {
                "epoch": epoch,
                "feature_extractor": feature_extractor.state_dict(),
                "label_predictor": label_predictor.state_dict(),
                "optimizer_F": optimizer_F.state_dict(),
                "optimizer_C": optimizer_C.state_dict(),
                "best_val_acc": best_val_acc,
            },
            f"{CHECKPOINT_DIR}/best_val_acc.pt",
        )
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            early_stop = True

    # 每 10 epoch 存一次 latest
    if (epoch + 1) % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "feature_extractor": feature_extractor.state_dict(),
                "label_predictor": label_predictor.state_dict(),
                "optimizer_F": optimizer_F.state_dict(),
                "optimizer_C": optimizer_C.state_dict(),
            },
            f"{CHECKPOINT_DIR}/latest.pt",
        )

    print(
        f"epoch {epoch:3d}: D_loss={train_D_loss:.4f}, F_loss={train_F_loss:.4f}, "
        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
    )
```

## 10. 校正

- 模型在訓練時可能存在預測偏好（例如某些類別預測機率偏高），導致預測分佈不均勻（Prior Shift）。
- 已知測試集（Target Domain）的類別分佈是完全均勻的分佈（每個類別 10%）。
- 校正邏輯：<my-orange>透過 logits - np.log(pred_dist)，將模型原本預測機率偏高的類別進行懲罰（扣分），預測機率偏低的類別進行補償，強行讓最終的預測結果逼近均勻分佈，這是比賽或實務中大幅提升 Test Accuracy 的強基線（Strong Baseline）後處理技術</my-orange>。

```bash
import numpy as np
import torch
from tqdm import tqdm


def calibrate_by_distribution(logits, eps=1e-8, target_dist=None):
    """
    利用測試集類別平衡做後處理校正
    logits: shape (N, 10) 的模型輸出 logits
    target_dist: 目標分佈，預設為均勻 [0.1, ..., 0.1]
    """
    if target_dist is None:
        target_dist = np.ones(10) / 10.0

    # 計算目前預測的平均機率分佈
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_dist = probs.mean(axis=0)

    # Prior Shift Correction：logits 減去 log(預測分佈)
    adjustment = np.log(pred_dist + eps)
    calibrated_logits = logits - adjustment

    # 取 argmax 得到最終預測
    calibrated_preds = np.argmax(calibrated_logits, axis=1)
    return calibrated_preds, pred_dist


# 若想更穩定，可加入以下版本，把上面 calibrate_by_distribution 換成 calibrate_iterative(all_logits) 即可。
def calibrate_iterative(logits, max_iter=5, eps=1e-8):
    """迭代式校正，直到分佈接近均勻"""
    current_logits = logits.copy().astype(np.float64)
    for _ in range(max_iter):
        probs = torch.softmax(torch.from_numpy(current_logits), dim=1).numpy()
        pred_dist = probs.mean(axis=0)
        adjustment = np.log(pred_dist + eps)
        current_logits = current_logits - adjustment
    return np.argmax(current_logits, axis=1)
```

## 11. Inference + Strong Baseline Calibration

```bash
label_predictor.eval()
feature_extractor.eval()

all_logits = []
all_probs = []

with torch.no_grad():
    for test_data, _ in tqdm(test_dataloader, desc="Inference"):
        test_data = test_data.to(device)
        features = feature_extractor(test_data)
        class_logits = label_predictor(features)

        all_logits.append(class_logits.cpu().numpy())

# 合併所有 logits
all_logits = np.concatenate(all_logits, axis=0)

# === 強基線：分佈校正 ===
final_preds, pred_dist = calibrate_by_distribution(all_logits)

print("預測類別分佈（校正前平均機率）:", np.round(pred_dist, 4))
print("校正後類別計數:", np.bincount(final_preds, minlength=10))

# === 輸出 submission ===
import pandas as pd

df = pd.DataFrame({"id": np.arange(len(final_preds)), "label": final_preds})
df.to_csv("DaNN_submission_strong_baseline.csv", index=False)
print("已輸出 DaNN_submission_strong_baseline.csv")
```

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
