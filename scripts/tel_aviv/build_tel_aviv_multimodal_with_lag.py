import os
import pandas as pd

RAW_DIR = r"D:\Morphology_Aware\data\raw\Tel_Aviv"
PROCESSED_DIR = r"D:\Morphology_Aware\data\processed\Tel_Aviv"
os.makedirs(PROCESSED_DIR, exist_ok=True)

BASE_FILE = fr"{PROCESSED_DIR}\tel_aviv_multimodal_final.csv"

# Choose target source
COUNTS_FILE = fr"{RAW_DIR}\tlv_60_sensors_SYNTHETIC_COUNTS.csv"
# Alternative if needed:
# COUNTS_FILE = fr"{RAW_DIR}\tlv_60_sensors_SYNTHETIC_DUBLIN_BASED.csv"

OUT_FILE = fr"{PROCESSED_DIR}\tel_aviv_multimodal_2023_WITH_TARGET_AND_LAG.csv"

TARGET = "volume_level"

def find_time_col(df, name):
    candidates = ["timestamp", "datetime", "date_time", "DateTime", "date", "time"]
    for c in candidates:
        if c in df.columns:
            print(f"{name} time column found: {c}")
            return c
    raise ValueError(f"No time column found in {name}. Columns: {df.columns.tolist()}")

def standardize_time(df, name):
    time_col = find_time_col(df, name)
    df = df.rename(columns={time_col: "timestamp"})
    df = df.loc[:, ~df.columns.duplicated()]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# -------------------------
# Load base multimodal file
# -------------------------
df = pd.read_csv(BASE_FILE)
df = standardize_time(df, "BASE")

print("BASE:", df.shape)
print("Base date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Base sensors:", df["sensor_id"].nunique())

# -------------------------
# Load target/counts file
# -------------------------
counts = pd.read_csv(COUNTS_FILE)
counts = standardize_time(counts, "COUNTS")

print("COUNTS:", counts.shape)
print("Counts date range:", counts["timestamp"].min(), "to", counts["timestamp"].max())
print("Counts sensors:", counts["sensor_id"].nunique())
print("Counts columns:", counts.columns.tolist())

# -------------------------
# Detect target column
# -------------------------
target_candidates = [
    "volume_level",
    "pedestrian_count",
    "count",
    "counts",
    "flow",
    "flow_count",
    "synthetic_count",
    "synthetic_counts",
    "estimated_flow",
    "proxy_flow"
]

found_target = None
for c in target_candidates:
    if c in counts.columns:
        found_target = c
        break

if found_target is None:
    raise ValueError(
        "No target column found in counts file. "
        f"Available columns: {counts.columns.tolist()}"
    )

print("Target column found:", found_target)

# Rename target to volume_level
if found_target != TARGET:
    counts = counts.rename(columns={found_target: TARGET})

# Keep only needed columns
counts = counts[["sensor_id", "timestamp", TARGET]].drop_duplicates(
    subset=["sensor_id", "timestamp"]
)

# -------------------------
# Merge target
# -------------------------
# Remove existing target if accidentally already present
df = df.drop(columns=[TARGET], errors="ignore")

df = df.merge(
    counts,
    on=["sensor_id", "timestamp"],
    how="left"
)

missing_target = df[TARGET].isna().mean()
print("Missing target ratio:", missing_target)

if missing_target > 0:
    print("WARNING: Some rows have missing target values.")
    print("Example missing rows:")
    print(df[df[TARGET].isna()][["sensor_id", "timestamp"]].head())

# -------------------------
# Clean duplicate weather cols
# -------------------------
df = df.drop(columns=[
    "temp_c_weather",
    "rel_humidity_weather",
    "Day_weather",
    "Month_weather",
    "Hour_weather",
    "Year_weather"
], errors="ignore")

# -------------------------
# Sort before lag creation
# -------------------------
df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

# -------------------------
# Build lag history
# -------------------------
df["lag1"] = df.groupby("sensor_id")[TARGET].shift(1)
df["lag_24"] = df.groupby("sensor_id")[TARGET].shift(24)
df["lag_168"] = df.groupby("sensor_id")[TARGET].shift(168)

df["is_peak_hour"] = df["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
df["lag1_x_peak"] = df["lag1"] * df["is_peak_hour"]

df[["lag1", "lag_24", "lag_168", "lag1_x_peak"]] = (
    df[["lag1", "lag_24", "lag_168", "lag1_x_peak"]].fillna(0)
)

# -------------------------
# Final validation
# -------------------------
print("\nFINAL:", df.shape)
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Sensors:", df["sensor_id"].nunique())
print("Timestamps:", df["timestamp"].nunique())
print("Duplicates:", df.duplicated().sum())

print("\nRows per sensor:")
print(df.groupby("sensor_id")["timestamp"].count().describe())

print("\nLag columns:")
print([c for c in df.columns if "lag" in c.lower()])

print("\nMissing values top 20:")
print(df.isnull().mean().sort_values(ascending=False).head(20))

print("\nFinal columns:")
print(df.columns.tolist())

# -------------------------
# Save
# -------------------------
df.to_csv(OUT_FILE, index=False)
print("\nSaved to:")
print(OUT_FILE)