## 1. Download Data 下載資料

- 使用 gdown (Google Drive) 下載 Kaggle 風格的 COVID-19 的訓練與測試集 CSV 檔案。
- 讀取本地資料。可改用 os.path 或 pathlib 寫相對路徑，或將路徑寫在 YAML 設定檔中。

```bash
# 1. Download Data
# tr_path = "covid.train.csv"
# tt_path = "covid.test.csv"

# 本地路徑
tr_path = r"C:\Users\User\Desktop\Miracle\Master\上\DL\HW\HW1\data\covid.train.csv"
tt_path = r"C:\Users\User\Desktop\Miracle\Master\上\DL\HW\HW1\data\covid.test.shuffle.csv"

# !gdown --id '19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF' --output covid.train.csv
# !gdown --id '1CE240jLm2npU-tdz81-oVKEF3T2yfT1O' --output covid.test.csv
```

## 2. Import Packages 匯入套件 + 設定隨機種子

- 設定隨機種子以確保每次跑的 Loss 和結果完全相同，才能<my-orange>客觀比較超參數</my-orange>。
- <my-cyan>cudnn.deterministic = True</my-cyan>：強制 NVIDIA GPU 使用固定的運算演算法，消除平行計算的浮點數微小誤差。
- <my-cyan>cudnn.benchmark = False</my-cyan>：關閉 cuDNN 的自動效能優化器，避免它在後台偷偷更換加速演算法。

```bash
# PyTorch，依序為:多維張量(Tensor)資料結構、神經網絡模塊、數據加載與預處理工具。
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def set_seed(seed=42069):
    torch.backends.cudnn.deterministic = True # 強制 NVIDIA cuDNN 使用「確定性（固定結果）」的卷積演算法，避免 GPU 平行運算時因浮點數誤差產生微小結果變動。
    torch.backends.cudnn.benchmark = False # 關閉 cuDNN 的自動演算法優化器。如果開啟（True），它會在後台不斷尋找最適合目前硬體的加速演算法，這會導致即使固定了亂數，每次跑的計算結果依然不同。
    np.random.seed(seed) # 固定 NumPy 套件的隨機數產生器種子，確保如矩陣打亂（Shuffle）或特定的資料預處理行為結果固定。
    torch.manual_seed(seed) # 固定 PyTorch 處理 CPU 計算時的隨機數種子，例如神經網路初始權重的亂數生成。
    if torch.cuda.is_available():
    # 檢查電腦是否有 NVIDIA GPU。若有，則固定所有 GPU 顯示卡上的隨機數種子，確保 GPU 上的隨機計算結果一致。
        torch.cuda.manual_seed_all(seed)
```

## 3. Some Utilities (定義工具函式 Utilities)

- 方便後續<my-orange>除錯與視覺化</my-orange>。
- get_device() 硬體分流：使用 <my-orange>torch.cuda.is_available() 自動偵測，有 GPU 走 cuda，沒有走 cpu</my-orange>。
- 多卡擴展性：<my-orange>若未來要用多張顯卡，需引入 PyTorch 的 DDP (分散式平行運算) 或使用 PyTorch Lightning 框架</my-orange>。

```bash
def get_device():
    # Get device (if GPU is available, use GPU)
    return 'cuda' if torch.cuda.is_available() else "cpu"

def plot_learning_curve(loss_record, title=""):
    # Plot learning curve of your DNN (train & dev loss)
    # 繪製訓練與驗證 loss 曲線
    ?...

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    # Plot prediction of your DNN
    # 繪製預測值 vs 真實值散點圖
    ?...

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c="r", alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c="b")
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel("ground truth value")
    plt.ylabel("predicted value")
    plt.title("Ground Truth v.s. Prediction")
    plt.show()
```

## 4. 建立 Dataset (資料集)

- 處理<my-orange>資料讀取與預處理</my-orange>。
- Mode 區分：<my-orange>train/dev 時必須同時回傳特徵 x 與標籤 y 用於計算損失；test 時沒有標籤，只回傳 x 進行預測</my-orange>。
- 重點：支援 target_only=True 做<my-orange>特徵選擇（40 states + 2 tested_positive）</my-orange>。
- target_only 特徵選擇：<my-orange>捨棄無用的各州類別特徵，只保留與未來高度相關的「歷史陽性率（tested_positive）」</my-orange>。
- 功用：大幅降低模型過擬合（Overfitting）風險，並加快模型收斂速度。

```bash
class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
    # 初始化與資料預處理
        # 讀取 CSV、特徵選擇、資料分割、標準化
        ?...

    def __getitem__(self, index):
    # 定義單筆資料的讀取規則
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
    # 回傳資料集的總筆數
        return len(self.data)
```

- 補：
    - <my-cyan>COVID19Dataset 在被建立出來時，就已經自己做完「標準化（Normalization）」與「訓練集/驗證集切分」了</my-cyan>。

```bash
if mode == "train":
    selected_indices = train_indices
else: # dev
    selected_indices = dev_indices
```

- 這代表當你執行 prep_dataloader(..., mode="train") 時，tr_set.dataset 裡面就已經只剩下 90% 的原始資料（另外 10% 被切去當作 dv_set 了）。
- 因此，當我們對 tr_set.dataset 跑 K-Fold 時，它的本質是：
    - <my-orange>把那 90% 的訓練資料，再切成 5 等分（每一折拿其中 4 份訓練、1 份驗證）</my-orange>。
    - <my-orange>它不會動到你原本預留的 10% 獨立驗證集（dv_set）</my-orange>，這在統計上也是完全正確且非常嚴謹的做法！

## 5. 建立 DataLoader

- 建立<my-orange>批次資料流</my-orange>。
- <my-cyan>shuffle=True 的時機</my-cyan>：<my-orange>只有訓練集（Train）需要打亂，目的是打破資料的時間或順序依賴，確保每個 Batch 數據多樣化。驗證與測試集絕對不打亂</my-orange>。
- <my-cyan>pin_memory=True 硬體加速</my-cyan>：在 CPU 記憶體中鎖定一塊區域（Page-locked），讓資料傳輸到 GPU 顯存時可以跳過記憶體分頁交換，利用 DMA（直接記憶體存取）高速傳輸，大幅減少資料載入的延遲。

```bash
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    # Generates a dataset, then is put into a dataloader.
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'),
        drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader
```

## 6. Deep Neural Network (定義神經網路 NeuralNet 模型)

- <my-cyan>Fully-Connected (全連接層)</my-cyan>：用來<my-orange>處理表格型數據（Tabular Data）</my-orange>的基礎架構。
- <my-cyan>BatchNorm1d (批次正規化)</my-cyan>：將每層輸入資料拉回標準正態分佈。功用：<my-orange>加速收斂、防止梯度消失（Gradient Vanishing）、允許使用較大學習率</my-orange>。
- <my-cyan>Dropout (隨機失活)</my-cyan>：訓練時隨機讓部分神經元斷開。功用：<my-orange>防止模型過擬合（Overfitting）</my-orange>，提高泛化能力。
- <my-cyan>L2 Regularization (L2 正規化)</my-cyan>：由 Optimizer 中的 weight_decay 參數處理，透過懲罰過大的權重來抑制過擬合。

```bash
# 6. Deep Neural Network
class NeuralNet(nn.Module):
    """A better fully-connected deep neural network"""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rates=[0.3, 0.3, 0.2]):
        super(NeuralNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        # L2 regularization is handled by optimizer (weight_decay)
        return self.criterion(pred, target)
```

## 7. DNN 訓練

- 包含 optimizer、loss、early stop。
- <my-cyan>Adam Optimizer</my-cyan>：結合動量與自適應學習率的優化器。功用：<my-orange>比傳統 SGD 收斂更快，適合初期的超參數探索</my-orange>。
- <my-cyan>Early Stopping (提前結束)</my-cyan>：監控 Validation loss（驗證集損失）。功用：<my-orange>當驗證集連續多個 Epoch（此處設定 80）沒有進步時，自動停止訓練</my-orange>，避免模型死背訓練資料（過擬合）。
- <my-cyan>model.train() 與 model.eval()</my-cyan>：必考！
    - train()：啟用 BatchNorm 的均值計算與 Dropout 的隨機丟棄。
    - eval()：固定 BatchNorm 的參數，並關閉 Dropout（讓所有神經元參與計算）。

```bash
def train(tr_set, dv_set, model, config, device):
    n_epochs = config["n_epochs"]  # Maximum number of epochs
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0

    while epoch < config['n_epochs']:
        model.train()  # set model to training mode
        for x, y in tr_set:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()  # compute gradient (backpropagation)
            optimizer.step()  # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print(
                "Saving model (epoch = {:4d}, loss = {:.4f})".format(epoch + 1, min_mse)
            )
            torch.save(
                model.state_dict(), config["save_path"]
            )  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record["dev"].append(dev_mse)
        if early_stop_cnt > config["early_stop"]:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print("Finished training after {} epochs".format(epoch))
    return min_mse, loss_record
```

## 8. Validation 訓練集 (驗證 + Early Stopping)

- <my-cyan>torch.no_grad()</my-cyan>：驗證與測試時使用。功用：關閉 PyTorch 的動態計算圖與梯度紀錄，大幅節省顯示記憶體（VRAM）並加速推理。

```bash
def dev(dv_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)  # compute averaged loss

    return total_loss
```

## 9. Testing 測試訓練集

```bash
def test(tt_set, model, device):
    model.eval()  # set model to evalutation mode
    preds = []
    for x in tt_set:  # iterate through the dataloader
        x = x.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            preds.append(pred.detach().cpu())  # collect prediction
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds
```

# 開始測試集

## 10. 定義 Setup Hyper-parameters

- 產生初始的 config 與 device = get_device()。

```bash
device = get_device()
os.makedirs("models", exist_ok=True)
target_only = True  # 使用 40 states + 2 tested_positive

config = {
    "n_epochs": 800,  # 足夠但不會過長
    "batch_size": 256,
    "optimizer": "Adam",  # 比 SGD 更快收斂
    "optim_hparas": {
        "lr": 0.0005,  # Adam 建議較小 lr
        "weight_decay": 1e-5,  # L2 正規化
    },
    "early_stop": 80,
    "save_path": "models/model.pth",
}
```

## 11. Load data and model 測試集

- 建立 tr_set, dv_set, tt_set 三個資料流。

```bash
tr_set = prep_dataloader(
    tr_path, "train", config["batch_size"], target_only=target_only
)
dv_set = prep_dataloader(tr_path, "dev", config["batch_size"], target_only=target_only)
tt_set = prep_dataloader(tt_path, "test", config["batch_size"], target_only=target_only)
```

## 12. Optuna 自動調參

- Optuna 搜尋空間：<my-orange>自動化尋找</my-orange>最佳學習率（lr）、權重衰減（weight_decay）、dropout_rate 與隱藏層維度（hidden_dim）。
- <my-cyan>MedianPruner (中位數剪枝)</my-cyan>：<my-orange>訓練到一半時，如果發現目前這組超參數的表現比之前歷史紀錄的中位數還要差，就直接提早砍掉該次訓練（Pruning），省下大量 GPU 算力與時間</my-orange>。
- <my-cyan>K-Fold (K折交叉驗證)</my-cyan>：<my-orange>將訓練集切成 K 份，輪流拿其中 1 份當驗證、K-1 份當訓練</my-orange>。功用：確保超參數在不同資料切分下都穩定，評估結果最客觀、最嚴謹。

```bash
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
import copy

def objective(trial, tr_set, dv_set, device):
    # 超參數搜尋空間
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])

    model = NeuralNet(
        tr_set.dataset.dim,
        hidden_dims=[hidden_dim, hidden_dim // 2, hidden_dim // 4],
        dropout_rates=[dropout_rate, dropout_rate, dropout_rate * 0.67]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 簡化訓練 loop（僅示範，實際可呼叫你原本的 train 函數）
    n_epochs = 50  # 先用較少 epoch 快速過濾
    for epoch in range(n_epochs):
        model.train()
        for x, y in tr_set:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = model.cal_loss(pred, y)
            loss.backward()
            optimizer.step()

        # 驗證
        val_loss = dev(dv_set, model, device)
        trial.report(val_loss, epoch)

        # === Optuna 剪枝機制 ===
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


def run_optuna_two_stage(tr_set, dv_set, device, n_trials=50, n_splits=5):
    # === 第一階段：單一驗證集 + 剪枝，快速找 Top-3 ===
    # 自動搜尋最佳超參數與進行 K 折交叉驗證的進階優化函式
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(lambda trial: objective(trial, tr_set, dv_set, device), n_trials=n_trials)

    # 取得前 3 名參數
    top_trials = sorted(study.trials, key=lambda t: t.value)[:3]
    print("Top 3 hyperparameters:")
    for i, t in enumerate(top_trials):
        print(f"Rank {i+1}: {t.params}, val_loss={t.value:.4f}")

    # === 第二階段：只對 Top-3 跑 K 折 ===
    best_kfold_models = []
    for rank, trial in enumerate(top_trials):
        print(f"\n=== Running K-Fold on Rank {rank+1} params ===")
        params = trial.params
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_models = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(tr_set.dataset)):
            # 以下根據 Dataset 自行修改
            model = NeuralNet(
				tr_set.dataset.dim,
				hidden_dims=[params['hidden_dim'], params['hidden_dim']//2, params['hidden_dim']//4],
				dropout_rates=[params['dropout_rate'], params['dropout_rate'], params['dropout_rate']*0.67]
			).to(device)
            # ... 訓練並儲存最佳 fold model
            fold_models.append(model)
        best_kfold_models.append(fold_models)

    return best_kfold_models, top_trials
```

- 呼叫方法

```bash
model = NeuralNet(tr_set.dataset.dim, hidden_dims=[64, 32, 16], dropout_rates=[0.3, 0.3, 0.2]).to(device)
```

```bash
est_models, top_params = run_optuna_two_stage(tr_set, dv_set, device, n_trials=30, n_splits=5)
```

- 輸出 Top 3 hyperparameters:

```bash
Rank 1: {'lr': 0.005468837472760657, 'weight_decay': 6.281986940525263e-05, 'dropout_rate': 0.4394394053719673, 'hidden_dim': 128}, val_loss=1.3305
Rank 2: {'lr': 0.005485265845460261, 'weight_decay': 0.0006598625544399672, 'dropout_rate': 0.4373530121245205, 'hidden_dim': 128}, val_loss=1.4290
Rank 3: {'lr': 0.00881509645435793, 'weight_decay': 0.00018335776011643492, 'dropout_rate': 0.41551504089830804, 'hidden_dim': 128}, val_loss=1.4306
```

# 13. Start Training 測試集

```bash
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
```

```bash
plot_learning_curve(model_loss_record, title="deep model")
```

```bash
del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config["save_path"], map_location="cpu")  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set
```

## 14.1 預測測試集並輸出 pred.csv

```bash
def save_pred(preds, file):
    # Save predictions to specified file
    print("Saving results to {}".format(file))
    with open(file, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "tested_positive"])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, "pred.csv")  # save prediction file to pred.csv
```

## 14.2 改使用 K-Fold 結果集成預測並輸出 pred_ensemble

- <my-cyan>Average Blending (算術平均集成)</my-cyan>：測試集預報時，不只用單一模型，而是讓 K-Fold 訓練出來的 K 個最優模型同時預測，並將結果取平均。
- 功用：<my-orange>有效降低預測的方差（Variance），在不增加額外訓練成本的前提下，直接穩定提升 Kaggle / 作業的盲測 RMSE 分數</my-orange>。

```bash
# 取得最優參數組合所訓練出的 5 個 K-Fold 模型
best_fold_models = est_models[0]

# 將 5 個模型全部切換為評估模式
for m in best_fold_models:
    m.eval()

ensemble_preds = []

# 遍歷測試集 DataLoader
for x in tt_set:
    x = x.to(device)
    fold_preds = []

    with torch.no_grad():
        # 讓 5 個模型同時對這批資料進行預測
        for m in best_fold_models:
            fold_preds.append(m(x).cpu())

    # 將 5 個模型的預測值取算術平均 (Average Blending)
    batch_avg_pred = torch.stack(fold_preds, dim=0).mean(dim=0)
    ensemble_preds.append(batch_avg_pred)

# 整合所有批次結果並轉為 NumPy
final_ensemble_preds = torch.cat(ensemble_preds, dim=0).numpy()
# 儲存集成預測結果
save_pred(final_ensemble_preds, "pred_ensemble.csv")
```

## 可優化

1. <my-orange>第一階段：用單一驗證集快速搜尋（含 Optuna MedianPruner 剪枝機制），找出 Top-3 參數組合。</my-orange>(已完成)
2. <my-orange>第二階段：只對 Top-3 跑 K 折交叉驗證。</my-orange>(已完成)
3. <my-orange>進階特徵工程</my-orange> (Feature Engineering) —— 投報率最高
    - 在這個 COVID-19 專案中，除了老師給的 42 維特徵（40州 + 2個陽性特徵），你可以在面試時主動提及你做了哪些特徵衍生：
    - 時間序列特徵（Lag Features）：確診率通常有週期性（例如週末篩檢人數少），可以加入「前一天、前兩天、上週同期的確診率差值（Differencing）」。
    - 移動平均（Rolling Statistics）：加入「過去 3 天、7 天的確診平均值或標準差」，能有效幫模型抹平數據中的隨機雜訊。
4. <my-orange>模型集成</my-orange> (Ensemble Learning) (已完成)
    - 當你用 Optuna 跑完 K 折交叉驗證後，你手頭上其實會擁有 K 個（例如 5 個）訓練好的最優模型權重。
    - 千萬不要只挑最強的那一個！
    - 正確做法（面試大加分）：在測試集（Test Set）進行推理時，讓這 5 個模型同時進行預測，並將 5 個預測值取算術平均（Average Blending）。這種 Ensemble 技巧不需要重新訓練，就能立刻降低預測的方差（Variance），在 RMSE 指標上通常能穩定再進步一小階。

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
