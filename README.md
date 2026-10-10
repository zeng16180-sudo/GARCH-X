# GARCH-X

## 模型設定

平均數方程式：

$$
r_t=\mu+\epsilon_t,\qquad \epsilon_t=\sqrt{h_t}z_t
$$

條件變異數方程式：

$$
h_t=\omega+\alpha\epsilon_{t-1}^2+\beta h_{t-1}
+\gamma\max(0,X_{t-1})
$$

| 參數 | 說明 |
|---|---|
| $\mu$ | 固定平均數 |
| $h_1$ | 初始條件變異數 |
| $\omega$ | 變異數方程的常數項 |
| $\alpha$ | 前一期衝擊平方的影響 |
| $\beta$ | 前一期條件變異數的影響 |
| $\gamma$ | 外生變數正值部分的影響 |

本版本使用前一期外生變數，並將其負值設為零。$h_t$ 是條件變異數，$\sqrt{h_t}$ 是條件波動率。

## 安裝

```bash
python -m pip install numpy pandas scipy matplotlib
```

## 專案結構

```text
garch-x/
├── garch_x.py
├── README.md
└── data/
    └── CWS1_data.csv
```

## 資料格式

CSV 第一欄為日期，並包含以下欄位：

| 欄位 | 說明 |
|---|---|
| `r_t` | 報酬率 |
| `IRD` | 外生變數，本範例使用利差 |

格式範例：

```csv
Date,r_t,IRD
2020/01/02,0.12,1.50
2020/01/03,-0.08,1.50
2020/01/06,0.05,1.55
```

報酬率與外生變數必須對應相同日期，且不得包含缺失值或無限值。

## 使用方式

### 1. 設定資料路徑

在 `garch_x.py` 中使用相對於程式位置的路徑：

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "CWS1_data.csv",
    index_col=0
)

df.index = pd.to_datetime(df.index, format="%Y/%m/%d")
df = df.sort_index()
```

如果第一筆報酬率因價格轉換而缺失，先略過該筆：

```python
df = df.iloc[1:]
```

只有在第一筆確實需要移除時才使用這行。

### 2. 選擇變數

```python
r_t = df["r_t"].to_numpy(dtype=float)
x = df["IRD"].to_numpy(dtype=float)
dates = df.index
```

使用其他外生變數時，修改欄位名稱：

```python
x = df["你的欄位名稱"].to_numpy(dtype=float)
```

### 3. 設定初始參數

```python
mu = 0.05
h1 = 0.05
alpha = 0.05
beta = 0.90
omega = 0.001
gamma = 0.01
```

初始值需符合：

```text
h1 > 0
omega > 0
alpha > 0
beta > 0
alpha + beta < 1
gamma >= 0
```

使用程式中的 `inverse_trans_param()`，將初始 `alpha`、`beta` 轉為最佳化參數：

```python
raw_total, raw_share = inverse_trans_param(1, alpha, beta)

init_params = (
    mu,
    h1,
    raw_total,
    raw_share,
    omega,
    gamma
)
```

## 注意事項

- 本版本限制 `gamma >= 0`，只估計外生變數正值部分增加變異數的效果。
- 若外生變數全部非正，經 `max(0, X)` 轉換後會全部為零，無法識別 `gamma`。
- 若直接使用帶正負號的外生變數，或允許負的 `gamma`，需重新設計變異數正值限制。
