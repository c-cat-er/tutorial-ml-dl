未完

## 1. Download Data 資料下載 與 環境準備

```bash
# Google Drive
!gdown --id '149paISvxCXDlr-720UEzseQA-y53rZxN' --output food-11.zip

# Dropbox
# !wget "https://www.dropbox.com/s/7yl5rra84ia0k8f/food-11.zip?dl=0" -O food-11.zip

# MEGA
# !wget https://megatools.megous.com/builds/megatools-1.10.3.tar.gz
# !tar -zxvf /content/megatools-1.10.3.tar.gz
# !sudo apt-get install libtool libglib2.0-dev gobject-introspection libgmp3-dev nettle-dev asciidoc glib-networking openssl libcurl4-openssl-dev libssl-dev
# %cd megatools-1.10.3/
# !./configure make
# !sudo make install
# %cd /content/
# !megadl 'https://mega.nz/file/FdlygByK#QQ5LP71MMjeXrXwoXM2qlygUsJ1D-6d5fMhj5gyi2Vc'

# Unzip the dataset.
!unzip food-11.zip
```

## 2. Import Packages 匯入套件

- torch.nn 包含<my-orange>建構神經網路必備的層（如 Linear, Conv2d）與損失函數（如 CrossEntropyLoss）</my-orange>。
- torchvision.transforms 負責將 <my-orange>PIL 影像格式轉為張量並做資料增強</my-orange>。
- ConcatDataset 用於<my-orange>將原始「有標籤訓練集」與篩選出來的「高信心偽標籤數據集」進行動態拼接，融合成新的訓練集</my-orange>。
- TensorDataset 用於<my-orange>將過濾出來的影像與標籤 Tensor 直接打包成 PyTorch 認可的 Dataset 格式，以便進行上述的拼接</my-orange>。

```bash
# PyTorch，第三個是為影像資料進行預處理與資料增強的模塊
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset, ConcatDataset
from torchvision.datasets import DatasetFolder

# for the progress bar.
from tqdm import tqdm
```

## 3. 資料增強 (Transforms) (data augmentation)

- <my-cyan>訓練集資料增強 (Data Augmentation) </my-cyan>：透過隨機旋轉、裁剪、顏色抖動（ColorJitter），提升模型的<my-orange>泛化</my-orange>能力（Generalization），防止 ResNet 等深層模型發生過擬合（Overfitting）。
- 測試與驗證集的處理限制：<my-orange>驗證/測試只做尺寸縮放(Resize) + 張量轉換(ToTensor)，不做任何隨機形變或色彩增強，以確保評估基準的穩定性與客觀性</my-orange>。
- <my-cyan>影像像素特徵轉換 (ToTensor)</my-cyan>：將 PIL Image 轉換為 PyTorch FloatTensor，自動將影像像素歸一化至 [0.0, 1.0]，有利於網路梯度收斂。

```bash
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
```

## 4. Dataset and DataLoader

- 同時保留 <my-orange>labeled 與 unlabeled 兩種資料集</my-orange>，為後續<my-orange>半監督學習</my-orange>做準備。
- <my-orange>新增 unlabeled_set_pseudo</my-orange>：用於<my-orange>預測偽標籤</my-orange>。

```bash
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm,)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm,)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm,)
unlabeled_set_pseudo = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm,  # 使用乾淨資料預測未知標籤)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm,)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

## 5. 建立模型 (Classifier)

- 直接使用 ResNet18 完整的架構（含卷積層）。(建議)

```bash
import torchvision.models as models
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 11)

    def forward(self, x):
        return self.model(x)
```

- 或改使用自定義 cnn_layers。

```bash
# import torchvision.models as models

# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__() # super(Classifier, self).__init__() 的簡化
#         # 使用 ResNet18，但不載入預訓練權重
#         self.model = models.resnet18(weights=None)          # 使用 ResNet18 骨幹

#         # input image size: [3, 128, 128]
#         # torch.nn.Conv2d(輸入通道數, 輸出通道數, 卷積核大小, 步長, 填充層數)
#         # torch.nn.MaxPool2d(池化核大小, 步長, 填充層數)
#         self.cnn_layers = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#             nn.Conv2d(128, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(4, 4, 0),
#         )

#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(in_features, 11) # 輸出 11 類

#         # 備用自訂 CNN（被註解掉）
#         self.cnn_layers = nn.Sequential(...)
#         self.fc_layers = nn.Sequential(...)

#     def forward(self, x):
#         x = self.model(x)   # 主要走 ResNet18
#         return x
#         ?
```

## 6. 半監督學習 (Pseudo-labeling)

- 半監督學習：<my-orange>先用少量的「有標籤資料」訓練模型，再用這個模型去預測「無標籤資料」，挑選高於門檻（如 threshold=0.85）的高信心樣本加入訓練集</my-orange>。
- 移除逐張 .append() 的 CPU-GPU 瓶頸，改用 batch slice + torch.cat。回傳 TensorDataset 可直接與 train_set Concat。
- 無標籤資料（Unlabeled Data）的雙重 Transforms 陷阱：
    - 預測階段（<my-orange>乾淨版本</my-orange>）：必須使用不做任何隨機形變的 test_tfm（如 unlabeled_set_pseudo），<my-orange>否則扭曲或裁剪的畫面會嚴重干擾模型的信心度判斷</my-orange>。
    - 訓練階段（<my-orange>強增強版本</my-orange>）：<my-orange>一旦樣本被選為偽標籤，併入訓練集時必須改用 train_tfm，藉由強增強來增加訓練難度、提升模型的泛化能力</my-orange>。

```bash
def get_pseudo_labels(dataset, model, threshold=0.85):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # 儲存符合條件的影像與偽標籤
    pseudo_indices_list, pseudo_labels_list = [], []
    sample_idx = 0

    for batch in tqdm(data_loader, desc="Pseudo-labeling"):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)
        max_probs, pred_labels = torch.max(probs, dim=1)

        # 過濾高信心樣本
        if mask.any():
            batch_size_actual = len(imgs)
            batch_indices = torch.arange(sample_idx, sample_idx + batch_size_actual, device="cpu")[mask.cpu()]
            pseudo_indices_list.append(batch_indices)
            pseudo_labels_list.append(pred_labels[mask].cpu())

        sample_idx += len(imgs)

    model.train()

    # 回傳符合高信心門檻的偽標籤資料集
    if len(pseudo_indices_list) == 0:
        print("No pseudo-labels generated (all below threshold).")
        return [], torch.tensor([])

    pseudo_indices = torch.cat(pseudo_indices_list, dim=0).tolist()
    pseudo_labels = torch.cat(pseudo_labels_list, dim=0)
    return pseudo_indices, pseudo_labels
```

- get_pseudo_labels_v2：用於訓練。
    - 用 unlabeled_set_pseudo（test_tfm）做穩定、可信的偽標籤；再用索引去抓 unlabeled_set（train_tfm），訓練時每次都會隨機強增強。這比較接近「弱視圖標註、強視圖訓練」的做法，泛化通常較好。

```bash
# get_pseudo_labels_v2（修正版：回傳索引 + 標籤，支援強增強）
def get_pseudo_labels_v2(dataset, model, threshold=0.85):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    pseudo_indices_list = []
    pseudo_labels_list = []
    sample_idx = 0   # 用來記錄目前在 dataset 中的全域索引

    for batch in tqdm(data_loader, desc="Pseudo-labeling"):
        imgs, _ = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        probs = softmax(logits)
        max_probs, pred_labels = torch.max(probs, dim=1)

        mask = max_probs > threshold
        if mask.any():
            # 計算這一批次中符合條件的樣本在原始 dataset 中的索引
            batch_size_actual = len(imgs)
            batch_indices = torch.arange(sample_idx, sample_idx + batch_size_actual, device="cpu")[mask.cpu()]
            pseudo_indices_list.append(batch_indices)
            pseudo_labels_list.append(pred_labels[mask].cpu())

        sample_idx += len(imgs)

    model.train()

    if len(pseudo_indices_list) == 0:
        print("No pseudo-labels generated (all below threshold).")
        return [], torch.tensor([])

    pseudo_indices = torch.cat(pseudo_indices_list, dim=0).tolist()   # Subset 需要 list
    pseudo_labels = torch.cat(pseudo_labels_list, dim=0)

    return pseudo_indices, pseudo_labels
```

## 7. 訓練迴圈（Train + Valid）(含半監督機制)

- AdamW：修正了傳統 Adam 在加入權重衰減（Weight Decay）時的數學錯誤。
- CosineAnnealingLR：使用餘弦函數來動態調整學習率，能讓模型在訓練後期更平穩地收斂。
- 偽標籤雙重機制：<my-orange>預測時用乾淨影像（確保信心度）；篩選出高信心樣本後，改用強增強訓練（提升泛化力）</my-orange>。
    - 用<my-orange>索引 (indices) 當橋樑</my-orange>。
    - <my-orange>get_pseudo_labels_v2 回傳「高信心樣本的索引列表」+「偽標籤」</my-orange>。
- <my-orange>標籤平滑化(Label Smoothing)</my-orange>(label_smoothing=0.1)：
    - 一種正規化技術。將非黑即白的一熱編碼（One-hot，如 [1, 0, 0]）轉化為輕微平滑的機率分佈（如 [0.9, 0.05, 0.05]）。
    - 防止模型對預測過度自信（Overconfidence），有效抑制過擬合（Overfitting）。

```bash
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

do_semi = True # 開啟半監督學習

for epoch in range(80):
    # 半監督：產生偽標籤。簡化後的 DataLoader 切換
    train_set_to_use = train_set
    if do_semi and epoch >= 5: # 前幾 epoch 先用有標籤資料穩定模型
        pseudo_indices, pseudo_labels = get_pseudo_labels(unlabeled_set_pseudo, model, threshold=0.85)
        if len(pseudo_indices) > 0:
            strong_pseudo_images_set = Subset(unlabeled_set, pseudo_indices)

            class PseudoDataset(torch.utils.data.Dataset):
                def __init__(self, subset_images, labels):
                    self.subset_images = subset_images
                    self.labels = labels
                def __len__(self):
                    return len(self.labels)
                def __getitem__(self, idx):
                    img, _ = self.subset_images[idx]
                    label = self.labels[idx]
                    return img, label

            pseudo_set = PseudoDataset(strong_pseudo_images_set, pseudo_labels)
            train_set_to_use = ConcatDataset([train_set, pseudo_set])

    train_loader = DataLoader(
        train_set_to_use, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # === Training ===
    model.train()
    train_loss = 0.0
    train_accs = []
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss += loss.item()
        train_accs.append(acc)

    train_loss = train_loss / len(train_loader)
    train_acc = sum(train_accs) / len(train_accs)
    scheduler.step()

    # === Validation ===
    model.eval()
    valid_loss = 0.0
    valid_accs = []
    with torch.no_grad():
        for batch in valid_loader:
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_loss += loss.item()
            valid_accs.append(acc)
    valid_loss = valid_loss / len(valid_loader)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"Epoch {epoch + 1:03d} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")
```

- 修改後的訓練迴圈 (Epoch 內)。

```bash
if do_semi and epoch >= 5:
    # 1. 依然使用「乾淨資料」送入模型預測偽標籤
    #    此處必須修改 get_pseudo_labels 函數，讓它改為回傳「高信心樣本的索引列表 (indices)」以及「對應的標籤」
    pseudo_indices, pseudo_labels = get_pseudo_labels_v2(unlabeled_set_pseudo, model, threshold=0.85)

    if len(pseudo_indices) > 0:
        # 2. 關鍵：使用「索引」去對應擁有強增強 (train_tfm) 的 unlabeled_set 取出子集
        from torch.utils.data import Subset
        strong_pseudo_images_set = Subset(unlabeled_set, pseudo_indices)

        # 3. 將強增強的影像子集 與 預測出來的標籤 重新打包
        #    自訂一個簡單的 Dataset 把 Subset 的影像與預測的 label 綁在一起
        class PseudoDataset(torch.utils.data.Dataset):
            def __init__(self, subset_images, labels):
                self.subset_images = subset_images
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                # 這裡會觸發 unlabeled_set 的 train_tfm，影像會被隨機變形！
                img, _ = self.subset_images[idx]
                label = self.labels[idx]
                return img, label

        pseudo_set = PseudoDataset(strong_pseudo_images_set, pseudo_labels)

        # 4. 融合原本的有標籤訓練集（此時偽標籤在每次 Epoch 訓練時都會被隨機強增強）
        train_set_to_use = ConcatDataset([train_set, pseudo_set])
```

## 8. 測試推論

```bash
# 像 Dropout 或 BatchNorm（批歸一化）這樣的模組，會根據模型目前是處於『訓練模式』還是『評估/測試模式』來改變運作方式。
model.eval()
# 建立一個空的列表，用來儲存模型對所有測試圖片的預測結果
predictions = []

# Iterate the testing set by batches.
# 使用迴圈，依據批次（Batch）逐一讀取測試資料集
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    # 每個批次包含「影像資料（imgs）」以及「對應的標籤（labels）」。
    # 但在這裡「labels」這個變數是完全沒用的，因為我們根本沒有測試集的真實答案（解答）。
    # 如果你把這些 labels 印出來，會發現它們全部都是 0。
    # 這是因為資料夾讀取器（DatasetFolder）規定一定要同時回傳每張圖片的影像和標籤，
    # 所以我們必須在讀取時建立「假的標籤（預設為 0）」來讓程式能正常運作。
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    # 在測試（推論）階段，我們不需要計算梯度（Gradient），而且我們也沒有真實標籤可以計算損失（Loss）。
    # 使用 torch.no_grad() 可以關閉梯度計算，進而大幅加快模型向前傳播（Forward）的速度、並節省記憶體。
    with torch.no_grad():
        # 將測試圖片送入 GPU/CPU 裝置，並輸入模型取得預測的分數（logits）
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    # 找出預測分數（logit）最大的那一類作為最終預測答案。
    # 接著將答案從 GPU 轉回 CPU、轉成 NumPy 陣列、再轉成 Python 列表，最後全部加進 predictions 列表中。
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
```

## 9. 輸出 CSV 儲存

```bash
with open("predict.csv", "w") as f:
    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    # 在 CSV 檔案中，除了第一行（標題行）之外，剩餘的每一行都代表一張圖片，並且每一行的圖片 ID（Image ID）都會對應一個模型預測出來的類別（Predicted Class）。
    for i, pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")
```

```bash
# 印出產生了多少偽標籤
print(f"Generated {len(pseudo_indices)} pseudo-labeled samples.")
```

##

- 關於偽標籤資料增強的額外建議
    - 預測階段已改用 unlabeled_set_pseudo（test_tfm）。
    - 若之後想讓偽標籤樣本在訓練時也套用 train_tfm，可改用「收集 indices + Subset(unlabeled_set_train, indices)」的方式（unlabeled_set_train 使用 train_tfm）

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
