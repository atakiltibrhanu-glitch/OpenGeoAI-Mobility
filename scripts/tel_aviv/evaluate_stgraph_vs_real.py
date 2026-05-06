import pandas as pd
import numpy as np

FILE = r"D:\Morphology_Aware\outputs\tel_aviv_stgraph\stgraph_transformer_predictions.csv"

df = pd.read_csv(FILE)

df = df.rename(columns={
    "y_true": "real",
    "y_pred": "pred"
})

print("\n=== BASIC ===")
print(df[["real", "pred"]].describe())

# ============================================
# Add synthetic time (approx for pattern check)
# ============================================

n = len(df)
df["hour"] = np.tile(np.arange(24), int(np.ceil(n/24)))[:n]

print("\n=== HOURLY PATTERN ===")
hour = df.groupby("hour")[["real", "pred"]].mean()
hour["diff"] = hour["real"] - hour["pred"]
print(hour)

# ============================================
# Weekend proxy (fake but ok for pattern)
# ============================================

df["is_weekend"] = (df.index % 7 >= 5).astype(int)

print("\n=== WEEKEND EFFECT ===")
print(df.groupby("is_weekend")[["real", "pred"]].mean())

# ============================================
# Accuracy metrics
# ============================================

from sklearn.metrics import accuracy_score, cohen_kappa_score

acc = accuracy_score(df["real"], df["pred"])
qwk = cohen_kappa_score(df["real"], df["pred"], weights="quadratic")
acc_pm1 = np.mean(np.abs(df["real"] - df["pred"]) <= 1)

print("\n=== FINAL METRICS ===")
print("Accuracy :", acc)
print("QWK      :", qwk)
print("Acc±1    :", acc_pm1)