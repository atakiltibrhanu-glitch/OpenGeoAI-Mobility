import pandas as pd
import numpy as np

FILE = r"D:\Morphology_Aware\outputs\tel_aviv_forecast\tel_aviv_forecast_2024_2025_FINAL_no_lag.csv"

df = pd.read_csv(FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["year"] = df["timestamp"].dt.year
df["hour"] = df["timestamp"].dt.hour
df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5,6])

# ============================================================
# BASIC
# ============================================================

print("\n=== BASIC ===")
print(df["predicted_volume_level"].describe())

# ============================================================
# YEAR COMPARISON
# ============================================================

print("\n=== BY YEAR ===")
print(df.groupby("year")["predicted_volume_level"].describe())

# ============================================================
# HOURLY PATTERN
# ============================================================

print("\n=== HOURLY PATTERN ===")
hour = df.groupby("hour")["predicted_volume_level"].mean()
print(hour)

# ============================================================
# WEEKEND EFFECT
# ============================================================

print("\n=== WEEKEND EFFECT ===")
print(df.groupby("is_weekend")["predicted_volume_level"].mean())

# ============================================================
# DISTRIBUTION
# ============================================================

print("\n=== DISTRIBUTION ===")
print(df["predicted_volume_level"].value_counts(normalize=True))