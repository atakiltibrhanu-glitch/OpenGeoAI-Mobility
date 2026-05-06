import pandas as pd

FILE = r"D:\Morphology_Aware\outputs\tel_aviv_ml_results\ml_predictions_regression_and_PLOS_NO_LAG.csv"

df = pd.read_csv(FILE, low_memory=False)

print("Loaded:", df.shape)
print("\nTasks:")
print(df["task"].value_counts(dropna=False))

# Keep rows with PLOS prediction
df = df[df["predicted_PLOS"].notna()].copy()

df["PLOS"] = pd.to_numeric(df["PLOS"], errors="coerce")
df["predicted_PLOS"] = pd.to_numeric(df["predicted_PLOS"], errors="coerce")
df = df.dropna(subset=["PLOS", "predicted_PLOS"])

print("\nAfter filtering:", df.shape)

real = df.groupby("sensor_id")["PLOS"].mean()
pred = df.groupby("sensor_id")["predicted_PLOS"].mean()

print("Sensors:", len(real))
print("Unique real:", real.nunique())
print("Unique pred:", pred.nunique())

print("\nSpearman:", real.corr(pred, method="spearman"))
print("Pearson:", real.corr(pred, method="pearson"))

sensor_check = pd.DataFrame({
    "real_mean_PLOS": real,
    "pred_mean_PLOS": pred
})
sensor_check["abs_diff"] = abs(sensor_check["real_mean_PLOS"] - sensor_check["pred_mean_PLOS"])

print("\nSensor-level error:")
print(sensor_check["abs_diff"].describe())

print("\nWorst 10 sensors:")
print(sensor_check.sort_values("abs_diff", ascending=False).head(10))

print("\n=== SPATIAL CORRELATION ===")

spearman = real.corr(pred, method="spearman")
pearson = real.corr(pred, method="pearson")

print("Spearman:", spearman)
print("Pearson :", pearson)