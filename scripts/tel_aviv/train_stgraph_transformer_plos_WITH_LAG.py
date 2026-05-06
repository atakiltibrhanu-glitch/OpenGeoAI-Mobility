import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)

# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tel_aviv_multimodal_2023_REALISTIC_WITH_TARGET_AND_LAG.csv"
ADJ_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tlv_sensor_knn_adjacency.npy"
OUT_DIR = r"D:\Morphology_Aware\outputs\tel_aviv_stgraph"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 24
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 6
LR = 5e-4
N_CLASSES = 6

D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.2

BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_stgraph_transformer_WITH_LAG.pt")
RESULTS_PATH = os.path.join(OUT_DIR, "stgraph_transformer_results_WITH_LAG.csv")
PRED_PATH = os.path.join(OUT_DIR, "stgraph_transformer_predictions_WITH_LAG.csv")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# WITH-LAG setting:
# Keep lag1, lag_24, lag_168, and lag1_x_peak.
df = df.drop(
    columns=[
        "temp_c_weather",
        "rel_humidity_weather",
        "Day_weather",
        "Month_weather",
        "Hour_weather",
        "Year_weather",
        "volume_level",
    ],
    errors="ignore",
)

lag_cols = ["lag1", "lag_24", "lag_168", "lag1_x_peak"]

for c in lag_cols:
    if c not in df.columns:
        raise ValueError(f"Missing lag column: {c}")
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

print("Mode: WITH LAG")
print("Lag columns used:", lag_cols)

# ============================================================
# TEMPORAL FEATURES
# ============================================================

df["Hour"] = df["timestamp"].dt.hour
df["Day"] = df["timestamp"].dt.day
df["Month"] = df["timestamp"].dt.month
df["DayOfWeek"] = df["timestamp"].dt.dayofweek
df["WeekOfYear"] = df["timestamp"].dt.isocalendar().week.astype(int)
df["Year"] = df["timestamp"].dt.year
df["is_weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
df["is_peak_hour"] = df["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

# Recompute lag1_x_peak safely
df["lag1_x_peak"] = df["lag1"] * df["is_peak_hour"]

def time_of_day(h):
    if h < 6:
        return "night"
    elif h < 12:
        return "morning"
    elif h < 18:
        return "afternoon"
    else:
        return "evening"

df["time_of_day"] = df["Hour"].apply(time_of_day)

if "temp_c" in df.columns:
    temp_std = df["temp_c"].std()
    if temp_std == 0 or pd.isna(temp_std):
        df["temp_c_z"] = 0.0
    else:
        df["temp_c_z"] = (df["temp_c"] - df["temp_c"].mean()) / temp_std

# ============================================================
# CREATE 6-CLASS PLOS TARGET
# ============================================================

thresholds = np.percentile(df["count"], [20, 40, 60, 80, 90])

def to_plos(x):
    if x <= thresholds[0]:
        return 0
    elif x <= thresholds[1]:
        return 1
    elif x <= thresholds[2]:
        return 2
    elif x <= thresholds[3]:
        return 3
    elif x <= thresholds[4]:
        return 4
    else:
        return 5

df["PLOS"] = df["count"].apply(to_plos).astype(int)

print("DATA:", df.shape)
print("Device:", DEVICE)
print("Sensors:", df["sensor_id"].nunique())
print("Timestamps:", df["timestamp"].nunique())
print("PLOS thresholds:", thresholds)
print("PLOS distribution:")
print(df["PLOS"].value_counts(normalize=True).sort_index())

# ============================================================
# ORDER SENSORS AND TIMES
# ============================================================

sensor_order = (
    df[["sensor_id", "latitude", "longitude"]]
    .drop_duplicates("sensor_id")
    .sort_values("sensor_id")["sensor_id"]
    .tolist()
)

time_order = sorted(df["timestamp"].unique())

N = len(sensor_order)
T_total = len(time_order)

sensor_to_idx = {s: i for i, s in enumerate(sensor_order)}
time_to_idx = {t: i for i, t in enumerate(time_order)}

print("N sensors:", N)
print("T timestamps:", T_total)

# ============================================================
# FEATURES
# ============================================================

drop_cols = [
    "count",
    "PLOS",
    "timestamp",
    "sensor_name",
    "city",
    "country",
]

feature_cols = [c for c in df.columns if c not in drop_cols]

X_raw = df[feature_cols].copy()
cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()

X_enc = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)
X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(0)
X_enc = X_enc.astype(np.float32)

feature_names = X_enc.columns.tolist()
F = len(feature_names)

print("Feature columns before encoding:", len(feature_cols))
print("Feature count after encoding:", F)
print("Categorical columns:", cat_cols)

print("\nFeature check — lag columns present?")
for c in lag_cols:
    matches = [col for col in feature_names if col == c or col.startswith(c + "_")]
    print(c, "YES" if len(matches) > 0 else "NO")

df_enc = pd.concat(
    [
        df[["sensor_id", "timestamp", "PLOS"]].reset_index(drop=True),
        X_enc.reset_index(drop=True),
    ],
    axis=1,
)

# ============================================================
# BUILD TENSOR [T, N, F] AND TARGET [T, N]
# ============================================================

X_tensor = np.zeros((T_total, N, F), dtype=np.float32)
Y_tensor = np.zeros((T_total, N), dtype=np.int64)

for row in df_enc.itertuples(index=False):
    sensor_id = row[0]
    timestamp = row[1]
    plos = row[2]
    features = np.array(row[3:], dtype=np.float32)

    ti = time_to_idx[timestamp]
    ni = sensor_to_idx[sensor_id]

    X_tensor[ti, ni, :] = features
    Y_tensor[ti, ni] = plos

mean = X_tensor.reshape(-1, F).mean(axis=0, keepdims=True)
std = X_tensor.reshape(-1, F).std(axis=0, keepdims=True)
std[std == 0] = 1.0

X_tensor = (X_tensor - mean.reshape(1, 1, F)) / std.reshape(1, 1, F)

# ============================================================
# TRAIN / VALIDATION / TEST TIME SPLIT
# ============================================================

time_arr = pd.to_datetime(pd.Series(time_order))

train_idx = np.where(time_arr < pd.Timestamp("2023-09-01"))[0]
val_idx = np.where(
    (time_arr >= pd.Timestamp("2023-09-01"))
    & (time_arr < pd.Timestamp("2023-10-01"))
)[0]
test_idx = np.where(time_arr >= pd.Timestamp("2023-10-01"))[0]

print("Train timestamps:", len(train_idx))
print("Val timestamps:", len(val_idx))
print("Test timestamps:", len(test_idx))

# ============================================================
# DATASET
# ============================================================

class STGraphDataset(Dataset):
    def __init__(self, X, Y, valid_indices, seq_len):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.indices = []

        for t in valid_indices:
            if t - seq_len >= 0:
                self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.X[t - self.seq_len:t]  # [T, N, F]
        y = self.Y[t]                   # [N]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

train_ds = STGraphDataset(X_tensor, Y_tensor, train_idx, SEQ_LEN)
val_ds = STGraphDataset(X_tensor, Y_tensor, val_idx, SEQ_LEN)
test_ds = STGraphDataset(X_tensor, Y_tensor, test_idx, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))
print("Test samples:", len(test_ds))

# ============================================================
# LOAD GRAPH
# ============================================================

A = np.load(ADJ_FILE).astype(np.float32)
A = torch.tensor(A, dtype=torch.float32).to(DEVICE)

print("Adjacency:", A.shape)

# ============================================================
# MODEL
# ============================================================

class STGraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_nodes,
        d_model=256,
        nhead=4,
        num_layers=3,
        dropout=0.2,
        n_classes=6,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.graph_linear = nn.Linear(d_model, d_model)
        self.spatial_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            activation="gelu",
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.temporal_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x, A):
        # x: [B, T, N, F]
        B, T, N, Fdim = x.shape

        x = self.input_proj(x)  # [B, T, N, D]

        spatial_out = []

        for t in range(T):
            xt = x[:, t, :, :]  # [B, N, D]

            attn_out, _ = self.spatial_attn(xt, xt, xt)

            graph_out = torch.einsum("ij,bjd->bid", A, attn_out)
            graph_out = self.graph_linear(graph_out)

            xt = self.spatial_norm(xt + self.dropout(graph_out))
            spatial_out.append(xt)

        x = torch.stack(spatial_out, dim=1)  # [B, T, N, D]

        # temporal modeling per node
        x = x.permute(0, 2, 1, 3)      # [B, N, T, D]
        x = x.reshape(B * N, T, -1)    # [B*N, T, D]

        x = self.temporal_encoder(x)
        x = self.temporal_norm(x)

        last = x[:, -1, :]             # [B*N, D]
        avg = x.mean(dim=1)            # [B*N, D]
        pooled = 0.7 * last + 0.3 * avg

        logits = self.fc(pooled)       # [B*N, C]
        logits = logits.reshape(B, N, -1)

        return logits

model = STGraphTransformer(
    input_dim=F,
    num_nodes=N,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    n_classes=N_CLASSES,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_loader(model, loader):
    model.eval()
    all_true = []
    all_pred = []
    losses = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb, A)

            loss = loss_fn(
                logits.reshape(-1, N_CLASSES),
                yb.reshape(-1),
            )

            pred = logits.argmax(dim=-1).cpu().numpy()

            all_pred.extend(pred.reshape(-1))
            all_true.extend(yb.cpu().numpy().reshape(-1))
            losses.append(loss.item())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_true, all_pred),
        "mae_class": mean_absolute_error(all_true, all_pred),
        "acc_pm1": float(np.mean(np.abs(all_true - all_pred) <= 1)),
        "qwk": cohen_kappa_score(all_true, all_pred, weights="quadratic"),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted"),
        "macro_f1": f1_score(all_true, all_pred, average="macro"),
    }

    return metrics, all_true, all_pred

# ============================================================
# TRAIN WITH EARLY STOPPING
# ============================================================

print("\nTraining ST-Graph Transformer WITH LAG...")

best_val_qwk = -1
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb, A)

        loss = loss_fn(
            logits.reshape(-1, N_CLASSES),
            yb.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_losses.append(loss.item())

    val_metrics, _, _ = evaluate_loader(model, val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"TrainLoss={np.mean(train_losses):.4f} "
        f"ValLoss={val_metrics['loss']:.4f} "
        f"ValAcc={val_metrics['accuracy']:.4f} "
        f"ValQWK={val_metrics['qwk']:.4f}"
    )

    if val_metrics["qwk"] > best_val_qwk:
        best_val_qwk = val_metrics["qwk"]
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("  Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# ============================================================
# TEST BEST MODEL
# ============================================================

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

test_metrics, y_true, y_pred = evaluate_loader(model, test_loader)

print("\n=== ST-GRAPH TRANSFORMER WITH LAG TEST RESULTS ===")
for k, v in test_metrics.items():
    print(f"{k}: {v}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=4))

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))))

# ============================================================
# SAVE RESULTS
# ============================================================

pd.DataFrame(
    [
        {
            "model": "ST-Graph Transformer WITH LAG",
            "setting": "WITH_LAG",
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "epochs_max": EPOCHS,
            "best_val_qwk": best_val_qwk,
            **test_metrics,
        }
    ]
).to_csv(RESULTS_PATH, index=False)

pd.DataFrame(
    {
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": np.abs(y_true - y_pred),
    }
).to_csv(PRED_PATH, index=False)

print("\nSaved results:", RESULTS_PATH)
print("Saved predictions:", PRED_PATH)
print("Saved best model:", BEST_MODEL_PATH)
