## 1. Download Data 下載資料

```bash
# Training progress bar
!pip install -q qqdm
```

```bash
# Downloading data
!gdown --id '1BazldgZLF-OH5kp5SXRKFUn3kn3ITVWx' --output data-bin.tar.xz
```

```bash
# 解壓縮
!tar Jxvf data-bin.tar.xz
!rm data-bin.tar.xz
```

## 2. Import Packages 匯入套件

- 同時固定了 random、numpy、torch 的隨機種子。
- cudnn.deterministic = True：關閉 CuDNN 的自動優化，確保相同輸入每次計算出的梯度和結果完全一致，這是論文實驗與 Baseline 比較的標準做法。

```bash
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans

from qqdm import qqdm, format_str
import pandas as pd

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # Torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(19530615)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## 3. Loading data

```bash
train = np.load("data-bin/trainingset.npy", allow_pickle=True)
test = np.load("data-bin/testingset.npy", allow_pickle=True)

print(train.shape)
print(test.shape)
```

## 4. 定義模型 Autoencoder - Models & loss

- 先寫最簡單的 conv_autoencoder，再擴充 VAE / ResNet。
- 一次定義多種 Autoencoder，方便後續切換實驗。
- 其他<my-cyan>模型</my-cyan>：
    - fcn_autoencoder：全連接版（baseline）。
    - conv_autoencoder (CNN)：
    - VAE (Variational Autoencoder) + loss_vae()：變分自編碼器。
        - reparametrize（重參數化技巧）：模型不直接輸出隱變量（Latent vector），而是輸出均值 mu 與方差 logvar。因為「從分佈抽樣」這個動作不可微，所以必須透過 mu + eps \* std（eps 為獨立的高斯雜訊）將隨機性移到輸入端，確保反向傳播（Backpropagation）可以順利計算梯度。
        - loss_vae（損失函數）：包含 MSE 損失（重建誤差）與 KLD 損失（KL 散度）。KLD 的數學公式是用來約束隱變量空間必須符合標準正態分佈 \(\mathcal{N}(0, I)\)，防止隱空間流形破碎，確保生成的連續性。
    - Resnet：使用 ResNet-18 做 encoder。
        - [:-1]：切除 ResNet-18 最後的分類層（Fully Connected Layer），只借用它強大的卷積特徵提取能力作為 Encoder。Decoder 則使用 nn.ConvTranspose2d（轉置卷積）進行上採樣還原。
- nn.BatchNorm2d：用於穩定深層網路的輸入分佈、加速收斂，並具有微小的正規化效果，防止過擬合。

```bash
class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3),)

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64 * 64 * 3),
            nn.Tanh(),)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),  # Medium: 移除此層（模型更小）
            # nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1), # Medium: 移除對應層
            # nn.BatchNorm2d(48),
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),)
        # self.enc_out_1 = nn.Sequential(
        #     nn.Conv2d(24, 48, 4, stride=2, padding=1),
        #     nn.ReLU(),)
        # self.enc_out_2 = nn.Sequential(
        #     nn.Conv2d(24, 48, 4, stride=2, padding=1),
        #     nn.ReLU(),)
        self.enc_out_1 = nn.Conv2d(24, 48, 4, stride=2, padding=1)
        self.enc_out_2 = nn.Conv2d(24, 48, 4, stride=2, padding=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),)

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def loss_vae(recon_x, x, mu, logvar, criterion):
    mse = criterion(recon_x, x)  # mse loss
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


class Resnet(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(Resnet, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = (
            fc_hidden1,
            fc_hidden2,
            CNN_embed_dim,)

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (
            (5, 5),
            (3, 3),
            (3, 3),
            (3, 3),)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),)  # 2d padding

        # encoding components
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)

        self.fc3_mu = nn.Linear(
            self.fc_hidden2, self.CNN_embed_dim) # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self.k4,
                stride=self.s4,
                padding=self.pd4,
            ),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),)
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=8,
                kernel_size=self.k3,
                stride=self.s3,
                padding=self.pd3,
            ),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),)

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=3,
                kernel_size=self.k2,
                stride=self.s2,
                padding=self.pd2,
            ),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),)  # y = (y1, y2, y3) \in [0 ,1]^3

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        if x.shape[0] > 1:
            x = self.bn1(self.fc1(x))
        else:
            x = self.fc1(x)
        x = self.relu(x)
        if x.shape[0] > 1:
            x = self.bn2(self.fc2(x))
        else:
            x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3_mu(x)
        return x

    def decode(self, z):
        if z.shape[0] > 1:
            x = self.relu(self.fc_bn4(self.fc4(z)))
            x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        else:
            x = self.relu(self.fc4(z))
            x = self.relu(self.fc5(x)).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=True)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)

        return x_reconst
```

## 5. 資料集 Dataset

- 自訂 TensorDataset，內建影像正規化（[-1, 1]）。
- tensors.permute(0, 3, 1, 2)：將<my-orange>圖片從 HWC（高、寬、通道）轉為 PyTorch 標準的 CHW 格式</my-orange>。
- 2.0 \* x / 255.0 - 1.0：將像素值縮放到 [-1, 1]。這是為了完美<my-orange>對齊模型最後一層的 nn.Tanh() 激活函數</my-orange>（Tanh 的輸出範圍也是 [-1, 1]）。如果這裡沒對齊，模型將無法收斂。

```bash
class CustomTensorDataset(TensorDataset):
    # TensorDataset with support of transforms.
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(lambda x: 2.0 * x / 255.0 - 1.0),
            	# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.tensors)
```

## 6. 訓練

- 模型訓練：無監督/半監督學習。訓練集裡面只有正常圖片。模型在這個區塊只學習「如何完美重建正常圖片」。
- 設定超參數、建立 DataLoader、選擇模型、執行訓練迴圈。
- 超參數與訓練迴圈。
- 使用 qqdm 即時顯示進度，並同時儲存 best / last 模型。

```bash
num_epochs = 100  # 延長訓練時間
batch_size = 128
learning_rate = 1e-4

# dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=batch_size)

# 模型與優化器
model_type = "cnn"
model_classes = {
    "resnet": Resnet(),
    "fcn": fcn_autoencoder(),
    "cnn": conv_autoencoder(),
    "vae": VAE(),}
model = model_classes[model_type].to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_loss = np.inf
model.train()

for epoch in range(num_epochs):
    tot_loss = []
    for data in train_dataloader:
        img = data.float().to(device)
        if model_type == "fcn":
            img = img.view(img.shape[0], -1)

        output = model(img)
        if model_type == "vae":
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        else:
            loss = criterion(output, img)

        tot_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model.state_dict(), f"best_model_{model_type}.pt")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {mean_loss:.4f}")

# 只在最後存一次
torch.save(model.state_dict(), f"last_model_{model_type}.pt")
```

## 7. 推論 Inference 與 輸出

- 載入模型、計算每張測試圖的 anomaly score，並輸出 CSV。
- 固定使用 reduction="none" 計算 pixel-wise loss，再轉 anomaly score。
- <my-cyan>異常分數</my-cyan>：
    - 當測試集出現異常圖片時，因為模型沒看過，解碼還原時的重建誤差（Reconstruction Error）會很大。
    - 數學上計算原圖與重建圖在每個像素、每個通道上的平方差加總，最後開根號（即 RMSE）。<my-orange>RMSE 數值越高，代表該圖片是異常的機率越高</my-orange>。

```bash
# 7. Inference
eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size)

eval_loss = nn.MSELoss(reduction="none")

# 正確載入與 model_type 一致的模型
checkpoint_path = f"last_model_{model_type}.pt"
model = model_classes[model_type].to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

out_file = "PREDICTION_FILE.csv"

anomality = []
with torch.no_grad():
    for data in test_dataloader:
        img = data.float().to(device)
        if model_type == "fcn":
            img = img.view(img.shape[0], -1)

        output = model(img)
        if model_type == "vae":
            output = output[0]

        if model_type == "fcn":
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)

anomality = torch.cat(anomality, dim=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=["Predicted"])
df.to_csv(out_file, index_label="Id")
```

```bash
anomality = list()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        if model_type in ["cnn", "vae", "resnet"]:
            img = data.float().cuda()
        elif model_type in ["fcn"]:
            img = data.float().cuda()
            img = img.view(img.shape[0], -1)
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type in ["cnn", "resnet", "fcn"]:
            output = output
        elif model_type in ["res_vae"]:
            output = output[0]
        elif model_type in ["vae"]:  # , 'vqvae'
            output = output[0]
        if model_type in ["fcn"]:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=["Predicted"])
df.to_csv(out_file, index_label="Id")
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
