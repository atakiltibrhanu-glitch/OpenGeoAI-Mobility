import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

DATA_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tel_aviv_multimodal_2023_REALISTIC_WITH_TARGET_AND_LAG.csv"
ADJ_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tlv_sensor_knn_adjacency.npy"
MODEL_PATH = r"D:\Morphology_Aware\outputs\tel_aviv_stgraph\best_stgraph_transformer.pt"

OUT_DIR = r"D:\Morphology_Aware\outputs\shap"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 24
N_CLASSES = 6
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.2

BACKGROUND_SAMPLES = 20
EXPLAIN_SAMPLES = 50
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# MODEL CLASS — SAME AS TRAINING
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
# SHAP WRAPPER
# ============================================================

class STGraphSHAPWrapper(nn.Module):
    """
    SHAP requires a model with one input.
    This wrapper fixes the adjacency matrix A and returns one scalar output per sample.

    Output explained:
    expected ordinal PLOS score averaged across sensors:
    E[PLOS] = sum_c p(c) * c
    """

    def __init__(self, model, A):
        super().__init__()
        self.model = model
        self.A = A

    def forward(self, x):
        logits = self.model(x, self.A)          # [B, N, C]
        probs = torch.softmax(logits, dim=-1)   # [B, N, C]

        class_values = torch.arange(
            N_CLASSES,
            dtype=torch.float32,
            device=probs.device
        )

        expected_plos = (probs * class_values).sum(dim=-1)  # [B, N]
        city_score = expected_plos.mean(dim=1)              # [B]

        return city_score.unsqueeze(1)                      # [B, 1]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Important: no-lag model setting + remove leakage/proxy target features
df = df.drop(columns=[
    "lag1",
    "lag_24",
    "lag_168",
    "lag1_x_peak",
    "temp_c_weather",
    "rel_humidity_weather",
    "Day_weather",
    "Month_weather",
    "Hour_weather",
    "Year_weather",
    "volume_level",   # critical: remove target proxy / leakage
], errors="ignore")

# Same temporal engineering as training
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
    std_temp = df["temp_c"].std()
    if std_temp == 0 or pd.isna(std_temp):
        std_temp = 1.0
    df["temp_c_z"] = (df["temp_c"] - df["temp_c"].mean()) / std_temp


# ============================================================
# CREATE PLOS TARGET — SAME LOGIC AS TRAINING
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


# ============================================================
# ORDER SENSORS AND TIMES
# ============================================================

sensor_order = (
    df[["sensor_id", "latitude", "longitude"]]
    .drop_duplicates("sensor_id")
    .sort_values("sensor_id")
    ["sensor_id"]
    .tolist()
)

time_order = sorted(df["timestamp"].unique())

N = len(sensor_order)
T_total = len(time_order)

sensor_to_idx = {s: i for i, s in enumerate(sensor_order)}
time_to_idx = {t: i for i, t in enumerate(time_order)}

print("Sensors:", N)
print("Timestamps:", T_total)


# ============================================================
# FEATURE ENCODING — SAME STYLE AS TRAINING
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

print("Feature count after encoding:", F)
print("Categorical columns:", cat_cols)


# ============================================================
# BUILD TENSOR [T, N, F]
# ============================================================

df_enc = pd.concat(
    [
        df[["sensor_id", "timestamp", "PLOS"]].reset_index(drop=True),
        X_enc.reset_index(drop=True),
    ],
    axis=1,
)

X_tensor_np = np.zeros((T_total, N, F), dtype=np.float32)
Y_tensor_np = np.zeros((T_total, N), dtype=np.int64)

for row in df_enc.itertuples(index=False):
    sensor_id = row[0]
    timestamp = row[1]
    plos = row[2]
    features = np.array(row[3:], dtype=np.float32)

    ti = time_to_idx[timestamp]
    ni = sensor_to_idx[sensor_id]

    X_tensor_np[ti, ni, :] = features
    Y_tensor_np[ti, ni] = plos

# Normalize same as training style
mean = X_tensor_np.reshape(-1, F).mean(axis=0, keepdims=True)
std = X_tensor_np.reshape(-1, F).std(axis=0, keepdims=True)
std[std == 0] = 1.0

X_tensor_np = (X_tensor_np - mean.reshape(1, 1, F)) / std.reshape(1, 1, F)


# ============================================================
# BUILD SEQUENCE SAMPLES [B, T, N, F]
# ============================================================

time_arr = pd.to_datetime(pd.Series(time_order))
test_idx = np.where(time_arr >= pd.Timestamp("2023-10-01"))[0]

valid_indices = [t for t in test_idx if t - SEQ_LEN >= 0]

print("Valid test sequence samples:", len(valid_indices))

sequence_samples = []

for t in valid_indices:
    x_seq = X_tensor_np[t - SEQ_LEN:t]   # [24, N, F]
    sequence_samples.append(x_seq)

sequence_samples = np.stack(sequence_samples, axis=0)  # [B, 24, N, F]

# Sample background and explanation sequences
rng = np.random.default_rng(RANDOM_SEED)

bg_n = min(BACKGROUND_SAMPLES, len(sequence_samples))
ex_n = min(EXPLAIN_SAMPLES, len(sequence_samples))

bg_idx = rng.choice(len(sequence_samples), size=bg_n, replace=False)
ex_idx = rng.choice(len(sequence_samples), size=ex_n, replace=False)

background = torch.tensor(sequence_samples[bg_idx], dtype=torch.float32).to(DEVICE)
explain_x = torch.tensor(sequence_samples[ex_idx], dtype=torch.float32).to(DEVICE)

print("Background shape:", background.shape)
print("Explain shape:", explain_x.shape)


# ============================================================
# LOAD ADJACENCY + MODEL
# ============================================================

A_np = np.load(ADJ_FILE).astype(np.float32)

if A_np.shape[0] != N:
    print("WARNING: adjacency size mismatch. Cropping to minimum.")
    min_n = min(A_np.shape[0], N)

    A_np = A_np[:min_n, :min_n]
    background = background[:, :, :min_n, :]
    explain_x = explain_x[:, :, :min_n, :]
    N = min_n

A = torch.tensor(A_np, dtype=torch.float32).to(DEVICE)

model = STGraphTransformer(
    input_dim=F,
    num_nodes=N,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    n_classes=N_CLASSES,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

wrapped_model = STGraphSHAPWrapper(model, A).to(DEVICE)
wrapped_model.eval()


# ============================================================
# RUN SHAP GRADIENT EXPLAINER
# ============================================================

print("Running SHAP GradientExplainer on trained ST-Graph model...")

explainer = shap.GradientExplainer(wrapped_model, background)

shap_values = explainer.shap_values(explain_x)

# Different SHAP versions may return list
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.array(shap_values)

# Expected shape: [B, T, N, F]
print("SHAP values shape:", shap_values.shape)

if shap_values.ndim == 5:
    # possible [B, T, N, F, 1]
    shap_values = shap_values.squeeze(-1)

if shap_values.shape[-1] != F:
    raise ValueError(
        f"Unexpected SHAP feature dimension. Got {shap_values.shape[-1]}, expected {F}."
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# Aggregate across samples, timesteps, and sensors
# ============================================================

importance = np.abs(shap_values).mean(axis=(0, 1, 2))

shap_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
}).sort_values("importance", ascending=False)

importance_path = os.path.join(OUT_DIR, "tel_aviv_stgraph_shap_importance.csv")
shap_df.to_csv(importance_path, index=False)

print("\nTop 30 ST-Graph SHAP features:")
print(shap_df.head(30))


# ============================================================
# CATEGORY IMPORTANCE
# ============================================================

def feature_category(name):
    n = name.lower()

    if any(k in n for k in ["hour", "day", "week", "month", "year", "time_of_day", "weekend", "peak"]):
        return "Temporal"
    if any(k in n for k in ["temp", "humidity", "weather", "precip"]):
        return "Environmental"
    if any(k in n for k in ["betweenness", "closeness", "highway", "latitude", "longitude"]):
        return "Spatial / Network"
    if any(k in n for k in ["ndvi", "canopy", "land_use"]):
        return "Urban morphology / greenness"
    if any(k in n for k in ["poi", "access", "transit", "service"]):
        return "Accessibility"
    if "sensor_id" in n:
        return "Sensor identity"
    return "Other"

shap_df["category"] = shap_df["feature"].apply(feature_category)

category_df = (
    shap_df.groupby("category")["importance"]
    .sum()
    .reset_index()
    .sort_values("importance", ascending=False)
)

category_path = os.path.join(OUT_DIR, "tel_aviv_stgraph_shap_category_importance.csv")
category_df.to_csv(category_path, index=False)

print("\nCategory importance:")
print(category_df)


# ============================================================
# PLOTS
# ============================================================

# 1. Bar plot: top 20 features
top_n = 20
top_df = shap_df.head(top_n).iloc[::-1]

plt.figure(figsize=(8, 7))
plt.barh(top_df["feature"], top_df["importance"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title("ST-Graph SHAP Global Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tel_aviv_stgraph_shap_bar_top20.png"), dpi=300)
plt.close()

# 2. Category plot
cat_plot = category_df.iloc[::-1]

plt.figure(figsize=(8, 5))
plt.barh(cat_plot["category"], cat_plot["importance"])
plt.xlabel("Aggregated mean |SHAP value|")
plt.ylabel("Feature category")
plt.title("ST-Graph SHAP Category Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tel_aviv_stgraph_shap_category_importance.png"), dpi=300)
plt.close()

# 3. Beeswarm-style SHAP summary
# Flatten [B, T, N, F] -> [B*T*N, F]
flat_shap = shap_values.reshape(-1, F)
flat_x = explain_x.detach().cpu().numpy().reshape(-1, F)

max_points = min(5000, flat_x.shape[0])
sample_idx = rng.choice(flat_x.shape[0], size=max_points, replace=False)

flat_shap_sample = flat_shap[sample_idx]
flat_x_sample = flat_x[sample_idx]

plt.figure()
shap.summary_plot(
    flat_shap_sample,
    flat_x_sample,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tel_aviv_stgraph_shap_beeswarm_top20.png"), dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# SAVE RAW SHAP SAMPLE ARRAYS OPTIONAL
# ============================================================

np.save(os.path.join(OUT_DIR, "tel_aviv_stgraph_shap_values.npy"), shap_values)

print("\nDONE.")
print("Saved feature importance:", importance_path)
print("Saved category importance:", category_path)
print("Saved figures to:", OUT_DIR)