import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

DATA_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tel_aviv_multimodal_2023_REALISTIC_WITH_TARGET_AND_LAG.csv"
ADJ_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tlv_sensor_knn_adjacency.npy"
MODEL_PATH = r"D:\Morphology_Aware\outputs\tel_aviv_stgraph\best_stgraph_transformer.pt"

OUT_DIR = r"D:\Morphology_Aware\outputs\tel_aviv_extra_experiments"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 24
BATCH_SIZE = 64
N_CLASSES = 6

D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.2

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


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
        B, T, N, Fdim = x.shape

        x = self.input_proj(x)

        spatial_out = []

        for t in range(T):
            xt = x[:, t, :, :]
            attn_out, _ = self.spatial_attn(xt, xt, xt)

            graph_out = torch.einsum("ij,bjd->bid", A, attn_out)
            graph_out = self.graph_linear(graph_out)

            xt = self.spatial_norm(xt + self.dropout(graph_out))
            spatial_out.append(xt)

        x = torch.stack(spatial_out, dim=1)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * N, T, -1)

        x = self.temporal_encoder(x)
        x = self.temporal_norm(x)

        last = x[:, -1, :]
        avg = x.mean(dim=1)
        pooled = 0.7 * last + 0.3 * avg

        logits = self.fc(pooled)
        logits = logits.reshape(B, N, -1)

        return logits


# ============================================================
# DATASET
# ============================================================

class STGraphDataset(Dataset):
    def __init__(self, X, Y, valid_indices, seq_len, time_order):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.time_order = time_order
        self.indices = []

        for t in valid_indices:
            if t - seq_len >= 0:
                self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.X[t - self.seq_len:t]
        y = self.Y[t]
        timestamp = self.time_order[t]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            str(timestamp),
        )


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

print("Loading data...")
df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# No-lag setting
df = df.drop(columns=[
    "lag1", "lag_24", "lag_168", "lag1_x_peak",
    "temp_c_weather", "rel_humidity_weather",
    "Day_weather", "Month_weather", "Hour_weather", "Year_weather",
    "volume_level"
], errors="ignore")

df["Hour"] = df["timestamp"].dt.hour
df["Day"] = df["timestamp"].dt.day
df["Month"] = df["timestamp"].dt.month
df["DayOfWeek"] = df["timestamp"].dt.dayofweek
df["WeekOfYear"] = df["timestamp"].dt.isocalendar().week.astype(int)
df["Year"] = df["timestamp"].dt.year
df["is_weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
df["is_peak_hour"] = df["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

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
    df["temp_c_z"] = (df["temp_c"] - df["temp_c"].mean()) / df["temp_c"].std()

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

sensor_meta = (
    df[["sensor_id", "latitude", "longitude"]]
    .drop_duplicates("sensor_id")
    .sort_values("sensor_id")
    .reset_index(drop=True)
)

sensor_order = sensor_meta["sensor_id"].tolist()
time_order = sorted(df["timestamp"].unique())

N = len(sensor_order)
T_total = len(time_order)

sensor_to_idx = {s: i for i, s in enumerate(sensor_order)}
time_to_idx = {t: i for i, t in enumerate(time_order)}

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

print("Sensors:", N)
print("Timestamps:", T_total)
print("Features:", F)
print("Categorical:", cat_cols)

df_enc = pd.concat(
    [
        df[["sensor_id", "timestamp", "PLOS", "count", "temp_c", "rel_humidity", "Hour", "DayOfWeek", "is_weekend", "is_peak_hour"]].reset_index(drop=True),
        X_enc.reset_index(drop=True),
    ],
    axis=1,
)

X_tensor = np.zeros((T_total, N, F), dtype=np.float32)
Y_tensor = np.zeros((T_total, N), dtype=np.int64)

meta_records = []

for row in df_enc.itertuples(index=False):
    sensor_id = row[0]
    timestamp = row[1]
    plos = row[2]
    count = row[3]
    temp_c = row[4]
    rel_humidity = row[5]
    hour = row[6]
    dayofweek = row[7]
    is_weekend = row[8]
    is_peak_hour = row[9]
    features = np.array(row[10:], dtype=np.float32)

    ti = time_to_idx[timestamp]
    ni = sensor_to_idx[sensor_id]

    X_tensor[ti, ni, :] = features
    Y_tensor[ti, ni] = plos

    meta_records.append({
        "timestamp": timestamp,
        "sensor_id": sensor_id,
        "time_idx": ti,
        "sensor_idx": ni,
        "count": count,
        "PLOS": plos,
        "temp_c": temp_c,
        "rel_humidity": rel_humidity,
        "Hour": hour,
        "DayOfWeek": dayofweek,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
    })

meta_df = pd.DataFrame(meta_records)

mean = X_tensor.reshape(-1, F).mean(axis=0, keepdims=True)
std = X_tensor.reshape(-1, F).std(axis=0, keepdims=True)
std[std == 0] = 1.0

X_tensor = (X_tensor - mean.reshape(1, 1, F)) / std.reshape(1, 1, F)

time_arr = pd.to_datetime(pd.Series(time_order))

train_idx = np.where(time_arr < pd.Timestamp("2023-09-01"))[0]
val_idx = np.where((time_arr >= pd.Timestamp("2023-09-01")) & (time_arr < pd.Timestamp("2023-10-01")))[0]
test_idx = np.where(time_arr >= pd.Timestamp("2023-10-01"))[0]

test_ds = STGraphDataset(X_tensor, Y_tensor, test_idx, SEQ_LEN, time_order)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

A_np = np.load(ADJ_FILE).astype(np.float32)

if A_np.shape[0] != N:
    print("WARNING: adjacency size mismatch. Cropping to minimum.")
    min_n = min(A_np.shape[0], N)
    A_np = A_np[:min_n, :min_n]
    X_tensor = X_tensor[:, :min_n, :]
    Y_tensor = Y_tensor[:, :min_n]
    sensor_order = sensor_order[:min_n]
    sensor_meta = sensor_meta.iloc[:min_n]
    N = min_n

A_full = torch.tensor(A_np, dtype=torch.float32).to(DEVICE)

model = STGraphTransformer(
    input_dim=F,
    num_nodes=N,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    n_classes=N_CLASSES,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "mae_class": mean_absolute_error(y_true, y_pred),
        "acc_pm1": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "bias": float(np.mean(y_pred - y_true)),
    }


def evaluate_model(A_eval, return_records=False):
    model.eval()
    all_true = []
    all_pred = []
    records = []

    with torch.no_grad():
        for xb, yb, ts_batch in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb, A_eval)
            pred = logits.argmax(dim=-1).cpu().numpy()
            true = yb.cpu().numpy()

            all_pred.extend(pred.reshape(-1))
            all_true.extend(true.reshape(-1))

            if return_records:
                B = pred.shape[0]
                for b in range(B):
                    timestamp = pd.to_datetime(ts_batch[b])
                    for n_idx, sensor_id in enumerate(sensor_order):
                        records.append({
                            "timestamp": timestamp,
                            "sensor_id": sensor_id,
                            "sensor_idx": n_idx,
                            "y_true": int(true[b, n_idx]),
                            "y_pred": int(pred[b, n_idx]),
                            "error": int(pred[b, n_idx]) - int(true[b, n_idx]),
                            "abs_error": abs(int(pred[b, n_idx]) - int(true[b, n_idx])),
                        })

    metrics = compute_metrics(all_true, all_pred)

    if return_records:
        return metrics, pd.DataFrame(records)

    return metrics


# ============================================================
# EXPERIMENT 1A: SPATIAL ABLATION
# ============================================================

print("\nRunning spatial ablation...")

A_identity = torch.eye(N, dtype=torch.float32).to(DEVICE)
A_zero = torch.zeros((N, N), dtype=torch.float32).to(DEVICE)

perm = np.random.permutation(N)
A_shuffled_np = A_np[perm][:, perm]
A_shuffled = torch.tensor(A_shuffled_np, dtype=torch.float32).to(DEVICE)

spatial_results = []

for name, A_eval in [
    ("Full adjacency", A_full),
    ("Identity adjacency", A_identity),
    ("Zero adjacency", A_zero),
    ("Shuffled adjacency", A_shuffled),
]:
    metrics = evaluate_model(A_eval)
    metrics["setting"] = name
    spatial_results.append(metrics)
    print(name, metrics)

spatial_df = pd.DataFrame(spatial_results)
spatial_path = os.path.join(OUT_DIR, "spatial_ablation_results.csv")
spatial_df.to_csv(spatial_path, index=False)


# ============================================================
# EXPERIMENT 1B: SENSOR SPARSITY STRESS TEST
# ============================================================

print("\nRunning sensor sparsity stress test...")

base_metrics, base_pred_df = evaluate_model(A_full, return_records=True)

# merge meta info
base_pred_df["timestamp"] = pd.to_datetime(base_pred_df["timestamp"])
meta_for_merge = meta_df[[
    "timestamp", "sensor_id", "temp_c", "rel_humidity",
    "Hour", "DayOfWeek", "is_weekend", "is_peak_hour"
]].drop_duplicates(["timestamp", "sensor_id"])

base_pred_df = base_pred_df.merge(meta_for_merge, on=["timestamp", "sensor_id"], how="left")

def propagate_predictions_from_observed(pred_df_time, observed_sensors, k=5):
    observed = pred_df_time[pred_df_time["sensor_id"].isin(observed_sensors)].copy()
    target = pred_df_time.copy()

    lat_all = sensor_meta.set_index("sensor_id")["latitude"].to_dict()
    lon_all = sensor_meta.set_index("sensor_id")["longitude"].to_dict()

    obs_ids = observed["sensor_id"].tolist()
    obs_pred = observed["y_pred"].values.astype(float)

    result_preds = []

    for _, row in target.iterrows():
        sid = row["sensor_id"]

        if sid in observed_sensors:
            result_preds.append(int(row["y_pred"]))
        else:
            lat0 = lat_all[sid]
            lon0 = lon_all[sid]

            distances = []
            for oid in obs_ids:
                d = np.sqrt((lat0 - lat_all[oid]) ** 2 + (lon0 - lon_all[oid]) ** 2)
                distances.append(d)

            distances = np.array(distances)
            nearest_idx = np.argsort(distances)[:k]
            nearest_d = distances[nearest_idx]

            weights = 1.0 / np.maximum(nearest_d, 1e-6) ** 2
            weights = weights / weights.sum()

            pred_score = np.sum(weights * obs_pred[nearest_idx])
            pred_label = int(np.clip(np.round(pred_score), 0, 5))
            result_preds.append(pred_label)

    return result_preds


availability_levels = [1.0, 0.75, 0.50, 0.30, 0.20, 0.10]
n_repeats = 10

sparsity_rows = []

unique_times = sorted(base_pred_df["timestamp"].unique())
all_sensors = np.array(sensor_order)

for availability in availability_levels:
    n_obs = max(1, int(round(N * availability)))

    for rep in range(n_repeats):
        rng = np.random.default_rng(RANDOM_SEED + rep + int(availability * 1000))
        observed_sensors = set(rng.choice(all_sensors, size=n_obs, replace=False))

        all_true = []
        all_pred_sparse = []

        for t in unique_times:
            pred_t = base_pred_df[base_pred_df["timestamp"] == t].copy()

            sparse_pred = propagate_predictions_from_observed(
                pred_t,
                observed_sensors=observed_sensors,
                k=5,
            )

            all_true.extend(pred_t["y_true"].values.tolist())
            all_pred_sparse.extend(sparse_pred)

        metrics = compute_metrics(all_true, all_pred_sparse)
        metrics["sensor_availability"] = availability
        metrics["n_observed_sensors"] = n_obs
        metrics["repeat"] = rep
        sparsity_rows.append(metrics)

        print(f"Availability {availability:.2f}, repeat {rep}:", metrics)

sparsity_df = pd.DataFrame(sparsity_rows)

sparsity_summary = (
    sparsity_df
    .groupby(["sensor_availability", "n_observed_sensors"])
    .agg({
        "accuracy": ["mean", "std"],
        "qwk": ["mean", "std"],
        "mae_class": ["mean", "std"],
        "bias": ["mean", "std"],
        "acc_pm1": ["mean", "std"],
    })
    .reset_index()
)

sparsity_summary.columns = [
    "_".join(col).strip("_") for col in sparsity_summary.columns.values
]

sparsity_raw_path = os.path.join(OUT_DIR, "sensor_sparsity_raw_results.csv")
sparsity_summary_path = os.path.join(OUT_DIR, "sensor_sparsity_results.csv")

sparsity_df.to_csv(sparsity_raw_path, index=False)
sparsity_summary.to_csv(sparsity_summary_path, index=False)


# ============================================================
# EXPERIMENT 2: SUSTAINABILITY-RELEVANT ERROR ANALYSIS
# ============================================================

print("\nRunning sustainability-relevant error analysis...")

pred_df = base_pred_df.copy()

conditions = []

def add_condition(name, mask):
    subset = pred_df[mask].copy()
    if len(subset) == 0:
        return

    metrics = compute_metrics(subset["y_true"], subset["y_pred"])
    metrics["condition"] = name
    metrics["samples"] = len(subset)
    conditions.append(metrics)

add_condition("Temperature <= 30C", pred_df["temp_c"] <= 30)
add_condition("Temperature > 30C", pred_df["temp_c"] > 30)

add_condition("Humidity <= 80%", pred_df["rel_humidity"] <= 80)
add_condition("Humidity > 80%", pred_df["rel_humidity"] > 80)

add_condition("Peak hours", pred_df["Hour"].isin([7, 8, 9, 16, 17, 18]))
add_condition("Off-peak hours", ~pred_df["Hour"].isin([7, 8, 9, 16, 17, 18]))

add_condition("Weekday", pred_df["is_weekend"] == 0)
add_condition("Weekend", pred_df["is_weekend"] == 1)

add_condition("Morning peak 07-09", pred_df["Hour"].isin([7, 8, 9]))
add_condition("Evening peak 16-18", pred_df["Hour"].isin([16, 17, 18]))
add_condition("Night 00-04", pred_df["Hour"].isin([0, 1, 2, 3, 4]))

sustain_df = pd.DataFrame(conditions)
sustain_path = os.path.join(OUT_DIR, "sustainability_error_results.csv")
sustain_df.to_csv(sustain_path, index=False)


# ============================================================
# FIGURES
# ============================================================

print("\nCreating figures...")

# Spatial ablation figure
plt.figure(figsize=(8, 5))
plt.bar(spatial_df["setting"], spatial_df["qwk"])
plt.ylabel("QWK")
plt.xlabel("Spatial setting")
plt.title("Spatial Ablation: Effect of Graph Structure")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_spatial_ablation_qwk.png"), dpi=300)
plt.close()

# Sensor sparsity curves
plt.figure(figsize=(8, 5))
plt.plot(
    sparsity_summary["sensor_availability"],
    sparsity_summary["accuracy_mean"],
    marker="o",
    label="Accuracy"
)
plt.plot(
    sparsity_summary["sensor_availability"],
    sparsity_summary["qwk_mean"],
    marker="o",
    label="QWK"
)
plt.xlabel("Sensor availability")
plt.ylabel("Metric")
plt.title("Sensor Sparsity Stress Test")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_sensor_sparsity_accuracy_qwk.png"), dpi=300)
plt.close()

# Bias vs sensor availability
plt.figure(figsize=(8, 5))
plt.plot(
    sparsity_summary["sensor_availability"],
    sparsity_summary["bias_mean"],
    marker="o",
)
plt.axhline(0, linestyle="--")
plt.xlabel("Sensor availability")
plt.ylabel("Bias: mean(predicted - observed)")
plt.title("Bias vs Sensor Availability")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_bias_vs_sensor_availability.png"), dpi=300)
plt.close()

# Sustainability error figure
plt.figure(figsize=(10, 5))
plt.bar(sustain_df["condition"], sustain_df["mae_class"])
plt.ylabel("MAE class")
plt.xlabel("Condition")
plt.title("Sustainability-Relevant Error Analysis")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_sustainability_error_mae.png"), dpi=300)
plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nDONE.")
print("Saved:", spatial_path)
print("Saved:", sparsity_summary_path)
print("Saved:", sustain_path)
print("Figures saved in:", OUT_DIR)

print("\nSpatial ablation:")
print(spatial_df)

print("\nSensor sparsity summary:")
print(sparsity_summary)

print("\nSustainability error analysis:")
print(sustain_df)