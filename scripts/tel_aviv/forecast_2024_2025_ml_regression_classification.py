import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not installed. Install with: pip install lightgbm")

# ============================================================
# PATHS
# ============================================================

DATA_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tel_aviv_multimodal_2023_REALISTIC_WITH_TARGET_AND_LAG.csv"
WEATHER_FILE = r"D:\Morphology_Aware\data\raw\Tel_Aviv\tel_aviv_weather_clean_2023_to_now.csv"
OUT_DIR = r"D:\Morphology_Aware\outputs\tel_aviv_forecast"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "forecast_2024_2025_ML_regression_and_PLOS.csv")
SUMMARY_FILE = os.path.join(OUT_DIR, "forecast_2024_2025_ML_summary.csv")
THRESHOLD_FILE = os.path.join(OUT_DIR, "forecast_2024_2025_PLOS_thresholds.json")

RANDOM_STATE = 42
TARGET_REG = "count"
TARGET_CLS = "PLOS"

LOS_LABELS = ["A", "B", "C", "D", "E", "F"]

LAG_COLS = ["lag1", "lag_24", "lag_168", "lag1_x_peak"]

DROP_ALWAYS = [
    TARGET_REG,
    TARGET_CLS,
    "PLOS_label",
    "volume_level",
    "timestamp",
    "sensor_name",
    "city",
    "country",
]

# ============================================================
# LOAD TRAIN DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("TRAIN DATA:", df.shape)
print("Train range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Sensors:", df["sensor_id"].nunique())

# Clean duplicate weather columns
df = df.drop(
    columns=[
        "temp_c_weather",
        "rel_humidity_weather",
        "Day_weather",
        "Month_weather",
        "Hour_weather",
        "Year_weather",
    ],
    errors="ignore",
)

# Deployment setting: remove lag/history
df = df.drop(columns=LAG_COLS, errors="ignore")
print("Mode: NO-LAG deployment forecast")

# ============================================================
# CREATE 6-CLASS PLOS FROM TRAINING COUNT
# ============================================================

thresholds = np.percentile(df[TARGET_REG], [20, 40, 60, 80, 90])

def count_to_plos_numeric(x):
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

df[TARGET_CLS] = df[TARGET_REG].apply(count_to_plos_numeric).astype(int)
df["PLOS_label"] = df[TARGET_CLS].map(lambda x: LOS_LABELS[x])

print("\nPLOS thresholds from full 2023 count:")
print({
    "20th": thresholds[0],
    "40th": thresholds[1],
    "60th": thresholds[2],
    "80th": thresholds[3],
    "90th": thresholds[4],
})

print("\nTraining PLOS distribution:")
print(df["PLOS_label"].value_counts(normalize=True).sort_index())

# ============================================================
# FEATURES
# ============================================================

feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]

X_raw = df[feature_cols].copy()
y_reg = df[TARGET_REG].copy()
y_cls = df[TARGET_CLS].copy()

cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()

print("\nCategorical columns:", cat_cols)
print("Feature columns before encoding:", len(feature_cols))

X = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)
train_cols = X.columns.tolist()

print("Feature count after encoding:", X.shape[1])

# ============================================================
# MODELS
# ============================================================

reg_models = {
    "XGBoost": XGBRegressor(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

cls_models = {
    "XGBoost": XGBClassifier(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

if HAS_LGBM:
    reg_models["LightGBM"] = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cls_models["LightGBM"] = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multiclass",
        num_class=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

# ============================================================
# TRAIN MODELS
# ============================================================

print("\nTraining regression models...")
for name, model in reg_models.items():
    print("Training regression:", name)
    model.fit(X, y_reg)

print("\nTraining classification models...")
for name, model in cls_models.items():
    print("Training classification:", name)
    model.fit(X, y_cls)

# ============================================================
# BUILD FUTURE 2024–2025 DATA
# ============================================================

weather = pd.read_csv(WEATHER_FILE)
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

weather = weather[
    ["timestamp", "temp_c", "rel_humidity", "precipitation", "wind_speed"]
].drop_duplicates("timestamp")

static_cols = [
    "sensor_id",
    "sensor_name",
    "latitude",
    "longitude",
    "betweenness",
    "closeness",
    "land_use",
    "highway",
    "sensor_canopy_pct",
    "sensor_ndvi_mean",
]
static_cols = [c for c in static_cols if c in df.columns]
sensors = df[static_cols].drop_duplicates("sensor_id")

future_hours = pd.date_range(
    start="2024-01-01 00:00:00",
    end="2025-12-31 23:00:00",
    freq="h",
)

future = (
    sensors.assign(key=1)
    .merge(pd.DataFrame({"timestamp": future_hours, "key": 1}), on="key")
    .drop(columns="key")
)

future["Hour"] = future["timestamp"].dt.hour
future["Day"] = future["timestamp"].dt.day
future["Month"] = future["timestamp"].dt.month
future["DayOfWeek"] = future["timestamp"].dt.dayofweek
future["WeekOfYear"] = future["timestamp"].dt.isocalendar().week.astype(int)
future["Year"] = future["timestamp"].dt.year
future["is_weekend"] = future["DayOfWeek"].isin([5, 6]).astype(int)
future["is_peak_hour"] = future["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

def time_of_day(hour):
    if hour < 6:
        return "night"
    elif hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"

future["time_of_day"] = future["Hour"].apply(time_of_day)

future = future.merge(weather, on="timestamp", how="left")

future["temp_c_z"] = (future["temp_c"] - df["temp_c"].mean()) / df["temp_c"].std()

print("\nFUTURE:", future.shape)
print("Future range:", future["timestamp"].min(), "to", future["timestamp"].max())
print("Future sensors:", future["sensor_id"].nunique())

print("\nFuture weather missing:")
print(future[["temp_c", "rel_humidity", "precipitation", "wind_speed"]].isnull().mean())

missing_features = [c for c in feature_cols if c not in future.columns]
if missing_features:
    raise ValueError(f"Missing future features: {missing_features}")

X_future_raw = future[feature_cols].copy()
X_future = pd.get_dummies(X_future_raw, columns=cat_cols, dummy_na=True)
X_future = X_future.reindex(columns=train_cols, fill_value=0)

# ============================================================
# PREDICT 2024–2025
# ============================================================

summary_rows = []

for name, model in reg_models.items():
    print("\nPredicting regression:", name)
    pred_count = model.predict(X_future)
    pred_count = np.maximum(pred_count, 0)

    future[f"pred_count_{name}"] = pred_count

    # Convert regression count to PLOS using 2023 thresholds
    future[f"pred_PLOS_from_count_{name}"] = future[f"pred_count_{name}"].apply(
        count_to_plos_numeric
    ).astype(int)
    future[f"pred_PLOS_from_count_label_{name}"] = future[f"pred_PLOS_from_count_{name}"].map(
        lambda x: LOS_LABELS[x]
    )

    summary_rows.append({
        "model": name,
        "task": "regression_count",
        "pred_mean": future[f"pred_count_{name}"].mean(),
        "pred_std": future[f"pred_count_{name}"].std(),
        "pred_min": future[f"pred_count_{name}"].min(),
        "pred_max": future[f"pred_count_{name}"].max(),
    })

for name, model in cls_models.items():
    print("\nPredicting classification:", name)
    pred_plos = model.predict(X_future).astype(int)

    future[f"pred_PLOS_{name}"] = pred_plos
    future[f"pred_PLOS_label_{name}"] = [LOS_LABELS[i] for i in pred_plos]

    dist = pd.Series(pred_plos).value_counts(normalize=True).sort_index()

    row = {
        "model": name,
        "task": "classification_PLOS_A_F",
        "pred_mean": pred_plos.mean(),
        "pred_std": pred_plos.std(),
        "pred_min": pred_plos.min(),
        "pred_max": pred_plos.max(),
    }

    for i, lab in enumerate(LOS_LABELS):
        row[f"class_{lab}_share"] = float(dist.get(i, 0.0))

    summary_rows.append(row)

# ============================================================
# SAVE OUTPUTS
# ============================================================

future.to_csv(OUT_FILE, index=False)

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_FILE, index=False)

with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
    json.dump(
        {
            "thresholds_count": thresholds.tolist(),
            "classes": {
                "A": "0-20%",
                "B": "20-40%",
                "C": "40-60%",
                "D": "60-80%",
                "E": "80-90%",
                "F": "90-100%",
            },
            "computed_from": "Full 2023 training count",
            "forecast_period": "2024-01-01 to 2025-12-31",
            "setting": "NO-LAG deployment",
        },
        f,
        indent=2,
    )

print("\nSaved forecast:")
print(OUT_FILE)

print("\nSaved summary:")
print(SUMMARY_FILE)

print("\nSaved thresholds:")
print(THRESHOLD_FILE)

print("\nSummary:")
print(summary)