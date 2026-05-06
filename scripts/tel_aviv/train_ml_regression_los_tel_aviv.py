import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("WARNING: LightGBM not installed. Install with: pip install lightgbm")


# ============================================================
# PATHS
# ============================================================

DATA_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tel_aviv_multimodal_2023_REALISTIC_WITH_TARGET_AND_LAG.csv"
OUT_DIR = r"D:\Morphology_Aware\outputs\tel_aviv_ml_results"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_REG = "count"
TARGET_LOS = "PLOS"

RANDOM_STATE = 42


# ============================================================
# SETTINGS
# ============================================================

# Change this to False for deployment / no-history experiment
USE_LAG = False

LAG_COLS = ["lag1", "lag_24", "lag_168", "lag1_x_peak"]

DROP_ALWAYS = [
    "timestamp",
    "sensor_name",
    "city",
    "country",
    "volume_level",
    "PLOS",
    "PLOS_label",
]

LOS_LABELS = ["A", "B", "C", "D", "E", "F"]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("DATA:", df.shape)
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Sensors:", df["sensor_id"].nunique())

# Clean duplicated weather columns if present
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

if not USE_LAG:
    df = df.drop(columns=LAG_COLS, errors="ignore")
    print("Mode: NO LAG / deployment setting")
else:
    print("Mode: WITH LAG / upper-bound setting")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_df = df[df["timestamp"] < "2023-10-01"].copy()
test_df = df[df["timestamp"] >= "2023-10-01"].copy()

print("Train:", train_df.shape)
print("Test :", test_df.shape)

if TARGET_REG not in train_df.columns:
    raise ValueError(f"Missing regression target: {TARGET_REG}")


# ============================================================
# CREATE 6-CLASS PLOS TARGET FROM TRAINING THRESHOLDS ONLY
# ============================================================

thresholds = np.percentile(train_df[TARGET_REG], [20, 40, 60, 80, 90])

print("\nPLOS thresholds from training count only:")
print({
    "20th": thresholds[0],
    "40th": thresholds[1],
    "60th": thresholds[2],
    "80th": thresholds[3],
    "90th": thresholds[4],
})

def count_to_plos_numeric(x):
    if x <= thresholds[0]:
        return 0  # A
    elif x <= thresholds[1]:
        return 1  # B
    elif x <= thresholds[2]:
        return 2  # C
    elif x <= thresholds[3]:
        return 3  # D
    elif x <= thresholds[4]:
        return 4  # E
    else:
        return 5  # F

train_df[TARGET_LOS] = train_df[TARGET_REG].apply(count_to_plos_numeric).astype(int)
test_df[TARGET_LOS] = test_df[TARGET_REG].apply(count_to_plos_numeric).astype(int)

train_df["PLOS_label"] = train_df[TARGET_LOS].map(lambda x: LOS_LABELS[x])
test_df["PLOS_label"] = test_df[TARGET_LOS].map(lambda x: LOS_LABELS[x])

print("\nTrain PLOS distribution:")
print(train_df["PLOS_label"].value_counts(normalize=True).sort_index())

print("\nTest PLOS distribution:")
print(test_df["PLOS_label"].value_counts(normalize=True).sort_index())


# ============================================================
# FEATURES
# ============================================================

drop_cols = DROP_ALWAYS + [TARGET_REG]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()

y_train_reg = train_df[TARGET_REG].copy()
y_test_reg = test_df[TARGET_REG].copy()

y_train_cls = train_df[TARGET_LOS].copy()
y_test_cls = test_df[TARGET_LOS].copy()

cat_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()

print("\nCategorical columns:", cat_cols)
print("Feature columns before encoding:", len(feature_cols))

X_train = pd.get_dummies(X_train_raw, columns=cat_cols, dummy_na=True)
X_test = pd.get_dummies(X_test_raw, columns=cat_cols, dummy_na=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Feature count after encoding:", X_train.shape[1])


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "MAE_class": mean_absolute_error(y_true, y_pred),
        "Acc_pm1": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "QWK": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted"),
    }


# ============================================================
# MODELS
# ============================================================

regression_models = {
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

classification_models = {
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

if HAS_LIGHTGBM:
    regression_models["LightGBM"] = LGBMRegressor(
        n_estimators=700,
        max_depth=-1,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    classification_models["LightGBM"] = LGBMClassifier(
        n_estimators=700,
        max_depth=-1,
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
# TRAIN REGRESSION MODELS
# ============================================================

all_results = []
prediction_outputs = []

print("\n" + "=" * 70)
print("REGRESSION: target = count")
print("=" * 70)

for model_name, model in regression_models.items():
    print(f"\nTraining regression model: {model_name}")

    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    metrics = regression_metrics(y_test_reg, y_pred)

    print(metrics)

    row = {
        "task": "regression_count",
        "model": model_name,
        "use_lag": USE_LAG,
        **metrics,
    }
    all_results.append(row)

    pred_df = test_df[["sensor_id", "timestamp", TARGET_REG]].copy()
    pred_df["model"] = model_name
    pred_df["task"] = "regression_count"
    pred_df["prediction_count"] = y_pred
    prediction_outputs.append(pred_df)


# ============================================================
# TRAIN LOS CLASSIFICATION MODELS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION: target = 6-class PLOS A-F")
print("=" * 70)

for model_name, model in classification_models.items():
    print(f"\nTraining classification model: {model_name}")

    model.fit(X_train, y_train_cls)
    y_pred = model.predict(X_test).astype(int)

    metrics = classification_metrics(y_test_cls.values, y_pred)

    print(metrics)

    print("\nClassification report:")
    print(
        classification_report(
            y_test_cls,
            y_pred,
            target_names=LOS_LABELS,
            digits=4,
        )
    )

    cm = confusion_matrix(y_test_cls, y_pred, labels=list(range(6)))
    print("\nConfusion matrix:")
    print(cm)

    row = {
        "task": "classification_PLOS_A_F",
        "model": model_name,
        "use_lag": USE_LAG,
        **metrics,
    }
    all_results.append(row)

    pred_df = test_df[["sensor_id", "timestamp", TARGET_REG, TARGET_LOS, "PLOS_label"]].copy()
    pred_df["model"] = model_name
    pred_df["task"] = "classification_PLOS_A_F"
    pred_df["predicted_PLOS"] = y_pred
    pred_df["predicted_PLOS_label"] = [LOS_LABELS[i] for i in y_pred]
    pred_df["abs_class_error"] = np.abs(pred_df[TARGET_LOS] - pred_df["predicted_PLOS"])
    prediction_outputs.append(pred_df)

    cm_df = pd.DataFrame(cm, index=LOS_LABELS, columns=LOS_LABELS)
    cm_path = os.path.join(
        OUT_DIR,
        f"confusion_matrix_{model_name}_PLOS_{'with_lag' if USE_LAG else 'no_lag'}.csv",
    )
    cm_df.to_csv(cm_path)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

suffix = "WITH_LAG" if USE_LAG else "NO_LAG"

results_path = os.path.join(OUT_DIR, f"ml_results_regression_and_PLOS_{suffix}.csv")
preds_path = os.path.join(OUT_DIR, f"ml_predictions_regression_and_PLOS_{suffix}.csv")
threshold_path = os.path.join(OUT_DIR, f"PLOS_thresholds_{suffix}.json")

results_df.to_csv(results_path, index=False)

if prediction_outputs:
    predictions_df = pd.concat(prediction_outputs, ignore_index=True, sort=False)
    predictions_df.to_csv(preds_path, index=False)

with open(threshold_path, "w", encoding="utf-8") as f:
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
            "computed_from": "Jan-Sep 2023 training count only",
            "use_lag": USE_LAG,
        },
        f,
        indent=2,
    )

print("\n" + "=" * 70)
print("SAVED OUTPUTS")
print("=" * 70)
print("Results:", results_path)
print("Predictions:", preds_path)
print("Thresholds:", threshold_path)

print("\nFinal results:")
print(results_df)