import pandas as pd

FILE = r"D:\Morphology_Aware\outputs\tel_aviv_ml_results\ml_predictions_regression_and_PLOS_NO_LAG.csv"

df = pd.read_csv(FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Filter classification only
df = df[df["task"] == "classification_PLOS_A_F"].copy()

print("Loaded:", df.shape)

# =====================================================
# 1. BASIC
# =====================================================
print("\n=== BASIC ===")
print(df[["PLOS", "predicted_PLOS"]].describe())

# =====================================================
# 2. HOURLY
# =====================================================
print("\n=== HOURLY PATTERN ===")

real_hour = df.groupby(df["timestamp"].dt.hour)["PLOS"].mean()
pred_hour = df.groupby(df["timestamp"].dt.hour)["predicted_PLOS"].mean()

hour_df = pd.DataFrame({
    "real": real_hour,
    "pred": pred_hour,
    "diff": real_hour - pred_hour
})

print(hour_df)

# =====================================================
# 3. WEEKEND
# =====================================================
print("\n=== WEEKEND EFFECT ===")

df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5,6])

week = df.groupby("is_weekend")[["PLOS", "predicted_PLOS"]].mean()
print(week)

# =====================================================
# 4. SPATIAL
# =====================================================
print("\n=== SPATIAL CONSISTENCY ===")

real_sensor = df.groupby("sensor_id")["PLOS"].mean()
pred_sensor = df.groupby("sensor_id")["predicted_PLOS"].mean()

corr = real_sensor.corr(pred_sensor, method="spearman")
print("Spearman correlation:", corr)

# =====================================================
# 5. ERROR
# =====================================================
print("\n=== SENSOR ERROR ===")

sensor_df = pd.DataFrame({
    "real": real_sensor,
    "pred": pred_sensor
})

sensor_df["abs_diff"] = abs(sensor_df["real"] - sensor_df["pred"])

print(sensor_df["abs_diff"].describe())
print(sensor_df.sort_values("abs_diff", ascending=False).head(10))