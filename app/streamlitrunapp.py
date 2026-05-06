# streamlitrunapp.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import folium
from streamlit_folium import st_folium


# ============================================================
# PATHS
# ============================================================

DATA_PATH = r"D:\Morphology_Aware\data\ui\tel_aviv_features.csv"
LOC_PATH = r"D:\Morphology_Aware\data\ui\tel_aviv_locations.csv"

MODEL_PATH = r"D:\Morphology_Aware\outputs\tel_aviv_stgraph\best_stgraph_transformer.pt"
ADJ_PATH = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tlv_sensor_knn_adjacency.npy"

UNSENSED_PATH = r"D:\Morphology_Aware\data\ui\tel_aviv_unsensed_locations.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 24
N_CLASSES = 6
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.2


# ============================================================
# MODEL CLASS
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
# LOAD FUNCTIONS
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    loc = pd.read_csv(LOC_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df, loc


@st.cache_resource
def load_adjacency():
    A = np.load(ADJ_PATH).astype(np.float32)
    return torch.tensor(A, dtype=torch.float32).to(DEVICE)


@st.cache_resource
def load_model(input_dim, num_nodes):
    model = STGraphTransformer(
        input_dim=input_dim,
        num_nodes=num_nodes,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        n_classes=N_CLASSES,
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# LOCATION FIX
# ============================================================

def fix_location_columns(dataframe):
    dataframe = dataframe.copy()

    if "lat" not in dataframe.columns and "latitude" in dataframe.columns:
        dataframe = dataframe.rename(columns={"latitude": "lat"})

    if "lon" not in dataframe.columns and "longitude" in dataframe.columns:
        dataframe = dataframe.rename(columns={"longitude": "lon"})

    return dataframe


# ============================================================
# FEATURE PREPARATION
# ============================================================

def add_time_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

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
        std = df["temp_c"].std()
        if std == 0 or pd.isna(std):
            std = 1.0
        df["temp_c_z"] = (df["temp_c"] - df["temp_c"].mean()) / std

    return df


def build_encoded_features(df):
    drop_cols = [
        "count",
        "pedestrian_count",
        "PLOS",
        "timestamp",
        "sensor_name",
        "city",
        "country",
        "lat",
        "lon",
        "latitude",
        "longitude",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_raw = df[feature_cols].copy()
    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()

    X_enc = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_enc = X_enc.astype(np.float32)

    return X_enc


def align_feature_count(X_enc, target_f):
    X = X_enc.copy()

    if X.shape[1] < target_f:
        missing = target_f - X.shape[1]
        for i in range(missing):
            X[f"missing_feature_{i}"] = 0.0

    if X.shape[1] > target_f:
        X = X.iloc[:, :target_f]

    return X.astype(np.float32)


def build_sequence_tensor(df, sensor_order, target_f):
    df = df.copy()
    df = add_time_features(df)

    latest_times = sorted(df["timestamp"].dropna().unique())

    if len(latest_times) < SEQ_LEN:
        st.error(f"Not enough timestamps. Need at least {SEQ_LEN}, found {len(latest_times)}.")
        st.stop()

    selected_times = latest_times[-SEQ_LEN:]

    df_seq = df[df["timestamp"].isin(selected_times)].copy()

    full_index = pd.MultiIndex.from_product(
        [selected_times, sensor_order],
        names=["timestamp", "sensor_id"]
    )

    df_seq = (
        df_seq.sort_values(["timestamp", "sensor_id"])
        .drop_duplicates(subset=["timestamp", "sensor_id"], keep="last")
    )

    df_seq = (
        df_seq.set_index(["timestamp", "sensor_id"])
        .reindex(full_index)
        .reset_index()
    )

    df_seq = df_seq.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    df_seq = df_seq.groupby("sensor_id", group_keys=False).apply(lambda g: g.ffill().bfill())
    df_seq = df_seq.fillna(0)

    X_enc = build_encoded_features(df_seq)
    X_enc = align_feature_count(X_enc, target_f)

    X_np = X_enc.values.astype(np.float32)

    T = SEQ_LEN
    N = len(sensor_order)
    F = target_f

    X_np = X_np.reshape(T, N, F)

    mean = X_np.reshape(-1, F).mean(axis=0, keepdims=True)
    std = X_np.reshape(-1, F).std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    X_np = (X_np - mean.reshape(1, 1, F)) / std.reshape(1, 1, F)

    X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    return X_tensor


# ============================================================
# SCENARIO HELPERS
# ============================================================

def project_latest_sequence_to_future(df_input, selected_datetime, scenario):
    df_scenario = df_input.copy()
    df_scenario["timestamp"] = pd.to_datetime(df_scenario["timestamp"], errors="coerce")

    latest_times = sorted(df_scenario["timestamp"].dropna().unique())[-SEQ_LEN:]
    mask_latest = df_scenario["timestamp"].isin(latest_times)

    future_times = pd.date_range(
        end=selected_datetime,
        periods=SEQ_LEN,
        freq="h"
    )

    time_map = dict(zip(latest_times, future_times))

    df_scenario.loc[mask_latest, "timestamp"] = (
        df_scenario.loc[mask_latest, "timestamp"].map(time_map)
    )

    df_scenario.loc[mask_latest, "Hour"] = df_scenario.loc[mask_latest, "timestamp"].dt.hour
    df_scenario.loc[mask_latest, "Day"] = df_scenario.loc[mask_latest, "timestamp"].dt.day
    df_scenario.loc[mask_latest, "Month"] = df_scenario.loc[mask_latest, "timestamp"].dt.month
    df_scenario.loc[mask_latest, "DayOfWeek"] = df_scenario.loc[mask_latest, "timestamp"].dt.dayofweek
    df_scenario.loc[mask_latest, "WeekOfYear"] = df_scenario.loc[mask_latest, "timestamp"].dt.isocalendar().week.astype(int)
    df_scenario.loc[mask_latest, "Year"] = df_scenario.loc[mask_latest, "timestamp"].dt.year
    df_scenario.loc[mask_latest, "is_weekend"] = df_scenario.loc[mask_latest, "DayOfWeek"].isin([5, 6]).astype(int)
    df_scenario.loc[mask_latest, "is_peak_hour"] = df_scenario.loc[mask_latest, "Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    if "hour_sin" in df_scenario.columns:
        df_scenario.loc[mask_latest, "hour_sin"] = np.sin(
            2 * np.pi * df_scenario.loc[mask_latest, "Hour"] / 24
        )

    if "hour_cos" in df_scenario.columns:
        df_scenario.loc[mask_latest, "hour_cos"] = np.cos(
            2 * np.pi * df_scenario.loc[mask_latest, "Hour"] / 24
        )

    if scenario == "+20% Demand":
        for col in ["poi_count_300m", "count", "pedestrian_count"]:
            if col in df_scenario.columns:
                df_scenario.loc[mask_latest, col] = df_scenario.loc[mask_latest, col] * 1.2

    if scenario == "Rain Event":
        for col in ["precip", "precipitation"]:
            if col in df_scenario.columns:
                df_scenario.loc[mask_latest, col] = 1.0

    if scenario == "Peak-Hour Stress":
        df_scenario.loc[mask_latest, "is_peak_hour"] = 1
        for col in ["count", "pedestrian_count"]:
            if col in df_scenario.columns:
                df_scenario.loc[mask_latest, col] = df_scenario.loc[mask_latest, col] * 1.15

    if scenario == "Hot Day":
        if "temp_c" in df_scenario.columns:
            df_scenario.loc[mask_latest, "temp_c"] = 35.0

    if scenario == "High Humidity":
        if "rel_humidity" in df_scenario.columns:
            df_scenario.loc[mask_latest, "rel_humidity"] = 85.0

    return df_scenario


# ============================================================
# UNSENSED FUNCTIONS
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def create_demo_unsensed_locations(loc, n_points=120):
    center_lat = loc["lat"].mean()
    center_lon = loc["lon"].mean()

    rng = np.random.default_rng(42)

    lat_noise = rng.normal(0, 0.018, n_points)
    lon_noise = rng.normal(0, 0.018, n_points)

    unsensed = pd.DataFrame({
        "unsensed_id": [f"U{i+1:03d}" for i in range(n_points)],
        "lat": center_lat + lat_noise,
        "lon": center_lon + lon_noise,
    })

    return unsensed


def load_unsensed_locations(loc, n_points):
    if os.path.exists(UNSENSED_PATH):
        unsensed = pd.read_csv(UNSENSED_PATH)
        unsensed = fix_location_columns(unsensed)

        if "unsensed_id" not in unsensed.columns:
            unsensed["unsensed_id"] = [f"U{i+1:03d}" for i in range(len(unsensed))]

        return unsensed

    return create_demo_unsensed_locations(loc, n_points=n_points)


def propagate_to_unsensed(sensor_pred_df, unsensed_df, k=5, power=2):
    sensed_lat = sensor_pred_df["lat"].values
    sensed_lon = sensor_pred_df["lon"].values
    sensed_plos = sensor_pred_df["PLOS"].values.astype(float)

    results = []

    for _, u in unsensed_df.iterrows():
        d = haversine_distance(
            u["lat"],
            u["lon"],
            sensed_lat,
            sensed_lon
        )

        nearest_idx = np.argsort(d)[:k]
        nearest_d = d[nearest_idx]

        weights = 1.0 / np.maximum(nearest_d, 1.0) ** power
        weights = weights / weights.sum()

        propagated_score = np.sum(weights * sensed_plos[nearest_idx])
        propagated_plos = int(np.clip(np.round(propagated_score), 0, 5))

        confidence = float(1.0 / (1.0 + nearest_d.mean() / 500.0))

        results.append({
            "unsensed_id": u["unsensed_id"],
            "lat": u["lat"],
            "lon": u["lon"],
            "PLOS": propagated_plos,
            "PLOS_score": propagated_score,
            "nearest_distance_m": float(nearest_d.min()),
            "mean_k_distance_m": float(nearest_d.mean()),
            "confidence": confidence,
            "method": "graph_propagation_knn_idw",
        })

    return pd.DataFrame(results)


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Urban Digital Twin — Tel Aviv",
    layout="wide"
)

st.title("🌍 Urban Digital Twin — Tel Aviv")
st.subheader("Future Scenario Simulation for Sensed and Unsensed PLOS + Pedestrian Count Prediction")


# ============================================================
# LOAD DATA
# ============================================================

df, loc = load_data()

df = fix_location_columns(df)
loc = fix_location_columns(loc)

if "lat" not in loc.columns or "lon" not in loc.columns:
    st.error("Location file must contain lat/lon or latitude/longitude columns.")
    st.write("Current location columns:", loc.columns.tolist())
    st.stop()

if "sensor_id" not in loc.columns:
    st.error("Location file must contain sensor_id column.")
    st.stop()

if "sensor_id" not in df.columns:
    st.error("Feature file must contain sensor_id column.")
    st.stop()

if "timestamp" not in df.columns:
    st.error("Feature file must contain timestamp column.")
    st.stop()


# ============================================================
# PLOS → PEDESTRIAN COUNT LOOKUP
# ============================================================

if "count" not in df.columns:
    st.error("Dataset must contain 'count' column for pedestrian count estimation.")
    st.stop()

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

df_lookup = df.copy()
df_lookup["PLOS_lookup"] = df_lookup["count"].apply(to_plos)

plos_count_lookup = (
    df_lookup.groupby("PLOS_lookup")["count"]
    .median()
    .to_dict()
)


# ============================================================
# SENSOR ORDER + ADJACENCY
# ============================================================

sensor_order = sorted(df["sensor_id"].dropna().unique().tolist())

A = load_adjacency()

if A.shape[0] != len(sensor_order):
    st.warning(
        f"Adjacency size {A.shape[0]} does not match sensor count {len(sensor_order)}. "
        "Using the minimum matching size."
    )
    min_n = min(A.shape[0], len(sensor_order))
    sensor_order = sensor_order[:min_n]
    A = A[:min_n, :min_n]

df = df[df["sensor_id"].isin(sensor_order)].copy()
loc = loc[loc["sensor_id"].isin(sensor_order)].copy()
loc = loc.drop_duplicates(subset=["sensor_id"]).reset_index(drop=True)

state_dict_preview = torch.load(MODEL_PATH, map_location="cpu")
TARGET_F = state_dict_preview["input_proj.weight"].shape[1]

model = load_model(TARGET_F, len(sensor_order))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📅 Future Forecast Date Settings")

year = st.sidebar.selectbox("Year", [2023, 2024, 2025, 2026], index=3)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=0)
day = st.sidebar.selectbox("Day", list(range(1, 32)), index=0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

try:
    selected_datetime = pd.Timestamp(year=year, month=month, day=day, hour=hour)
except ValueError:
    st.sidebar.error("Invalid date selected. Please choose a valid day for the selected month.")
    st.stop()

day_type = "Weekend" if selected_datetime.dayofweek in [5, 6] else "Weekday"

st.sidebar.write(f"Selected datetime: {selected_datetime}")
st.sidebar.write(f"Day type: {day_type}")

st.sidebar.header("📍 Sensor Selection")

sensor_options = ["All Sensors"] + sorted(sensor_order)

selected_sensor = st.sidebar.selectbox(
    "Select Sensor ID",
    sensor_options
)

st.sidebar.header("📊 Scenario Settings")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Normal", "+20% Demand", "Rain Event", "Peak-Hour Stress", "Hot Day", "High Humidity"]
)

show_unsensed = st.sidebar.checkbox("Show Unsensed Locations", value=True)

k_neighbors = st.sidebar.slider(
    "Graph propagation neighbors",
    min_value=1,
    max_value=10,
    value=5
)

unsensed_points = st.sidebar.slider(
    "Demo unsensed points",
    min_value=20,
    max_value=300,
    value=120,
    step=20
)


# ============================================================
# MAIN PREDICTION
# ============================================================

df_scenario = project_latest_sequence_to_future(df, selected_datetime, scenario)

X_tensor = build_sequence_tensor(df_scenario, sensor_order, TARGET_F)

with torch.no_grad():
    logits = model(X_tensor, A)
    probs = torch.softmax(logits, dim=-1)

    pred = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy()
    conf = probs.max(dim=-1).values.squeeze(0).detach().cpu().numpy()

sensor_pred = pd.DataFrame({
    "sensor_id": sensor_order,
    "PLOS": pred.astype(int),
    "confidence": conf.astype(float),
    "selected_datetime": selected_datetime,
    "scenario": scenario,
})

sensor_pred["predicted_count"] = (
    sensor_pred["PLOS"].map(plos_count_lookup).fillna(0).round().astype(int)
)

sensor_map_all = sensor_pred.merge(loc, on="sensor_id", how="left")

if "lat" not in sensor_map_all.columns or "lon" not in sensor_map_all.columns:
    st.error("After merging predictions with locations, lat/lon columns are missing.")
    st.write("sensor_map columns:", sensor_map_all.columns.tolist())
    st.write("loc columns:", loc.columns.tolist())
    st.stop()

missing_coords = sensor_map_all[["lat", "lon"]].isna().sum().sum()

if missing_coords > 0:
    st.warning(f"{missing_coords} missing coordinate values found after merge.")

sensor_map_all = sensor_map_all.dropna(subset=["lat", "lon"]).reset_index(drop=True)

if len(sensor_map_all) == 0:
    st.error("No valid sensor coordinates available for mapping.")
    st.stop()


# ============================================================
# UNSENSED GRAPH PROPAGATION
# ============================================================

unsensed_df = load_unsensed_locations(loc, n_points=unsensed_points)

if "lat" not in unsensed_df.columns or "lon" not in unsensed_df.columns:
    st.error("Unsensed file must contain lat/lon or latitude/longitude columns.")
    st.write("Unsensed columns:", unsensed_df.columns.tolist())
    st.stop()

unsensed_pred_all = propagate_to_unsensed(
    sensor_pred_df=sensor_map_all,
    unsensed_df=unsensed_df,
    k=k_neighbors,
    power=2,
)

unsensed_pred_all["selected_datetime"] = selected_datetime
unsensed_pred_all["scenario"] = scenario

unsensed_pred_all["predicted_count"] = (
    unsensed_pred_all["PLOS"].map(plos_count_lookup).fillna(0).round().astype(int)
)


# ============================================================
# FILTER FOR SELECTED SENSOR
# ============================================================

sensor_map = sensor_map_all.copy()
unsensed_pred = unsensed_pred_all.copy()

if selected_sensor != "All Sensors":
    sensor_map = sensor_map[sensor_map["sensor_id"] == selected_sensor].copy()

    selected_loc = loc[loc["sensor_id"] == selected_sensor]

    if len(selected_loc) > 0:
        lat0 = selected_loc.iloc[0]["lat"]
        lon0 = selected_loc.iloc[0]["lon"]

        unsensed_pred["dist_to_selected_sensor_m"] = haversine_distance(
            unsensed_pred["lat"],
            unsensed_pred["lon"],
            lat0,
            lon0
        )

        unsensed_pred = (
            unsensed_pred.sort_values("dist_to_selected_sensor_m")
            .head(30)
            .reset_index(drop=True)
        )


# ============================================================
# MAP
# ============================================================

st.subheader("🗺️ Future PLOS + Pedestrian Count Map: Sensed + Unsensed Locations")

st.info(
    f"Scenario-based simulation for {selected_datetime} | "
    f"{day_type} | Scenario: {scenario} | Sensor: {selected_sensor}"
)

center_lat = sensor_map_all["lat"].mean()
center_lon = sensor_map_all["lon"].mean()

if selected_sensor != "All Sensors" and len(sensor_map) > 0:
    center_lat = sensor_map.iloc[0]["lat"]
    center_lon = sensor_map.iloc[0]["lon"]

m = folium.Map(location=[center_lat, center_lon], zoom_start=13 if selected_sensor != "All Sensors" else 12)


def get_color(plos):
    colors = ["green", "lightgreen", "yellow", "orange", "red", "darkred"]
    return colors[int(plos)]


sensed_layer = folium.FeatureGroup(name="Sensed Sensor Predictions", show=True)

for _, row in sensor_map.iterrows():
    radius = 13 if row["sensor_id"] == selected_sensor else 8

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=get_color(row["PLOS"]),
        fill=True,
        fill_opacity=0.95,
        popup=(
            f"<b>Sensed Sensor</b><br>"
            f"Sensor ID: {row['sensor_id']}<br>"
            f"PLOS: {int(row['PLOS'])}<br>"
            f"Pedestrian Count: {int(row['predicted_count'])}<br>"
            f"Confidence: {row['confidence']:.3f}<br>"
            f"Date: {selected_datetime}<br>"
            f"Scenario: {scenario}"
        ),
    ).add_to(sensed_layer)

sensed_layer.add_to(m)

if show_unsensed:
    unsensed_layer = folium.FeatureGroup(name="Unsensed Graph-Propagated Predictions", show=True)

    for _, row in unsensed_pred.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=get_color(row["PLOS"]),
            fill=True,
            fill_opacity=0.55,
            popup=(
                f"<b>Unsensed Location</b><br>"
                f"ID: {row['unsensed_id']}<br>"
                f"Propagated PLOS: {int(row['PLOS'])}<br>"
                f"Estimated Count: {int(row['predicted_count'])}<br>"
                f"PLOS score: {row['PLOS_score']:.2f}<br>"
                f"Nearest sensor: {row['nearest_distance_m']:.1f} m<br>"
                f"Confidence: {row['confidence']:.3f}<br>"
                f"Date: {selected_datetime}<br>"
                f"Method: graph propagation"
            ),
        ).add_to(unsensed_layer)

    unsensed_layer.add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, width=1100, height=600)


# ============================================================
# SELECTED SENSOR DETAILS
# ============================================================

st.subheader("📍 Selected Sensor Details")

if selected_sensor != "All Sensors" and len(sensor_map) > 0:
    sensor_info = sensor_map.iloc[0]

    c1,  c3,  c5 = st.columns(3)

    c1.metric("Sensor ID", sensor_info["sensor_id"])
    
    c3.metric("Pedestrian Count", int(sensor_info["predicted_count"]))
    
    c5.metric("Nearby Unsensed Points", len(unsensed_pred))

    st.write(
        f"Location: ({sensor_info['lat']:.5f}, {sensor_info['lon']:.5f})"
    )
else:
    st.info("Select one Sensor ID from the sidebar to view detailed sensor information.")


# ============================================================
# TIME-SERIES PREDICTION PANEL FOR SELECTED SENSOR
# ============================================================

st.subheader("⏱️ Selected Sensor 24-Hour Future PLOS + Pedestrian Count Profile")

if selected_sensor == "All Sensors":
    st.info("Select one Sensor ID from the sidebar to view its 24-hour prediction profile.")
else:
    future_hours = pd.date_range(
        start=selected_datetime.normalize(),
        periods=24,
        freq="h"
    )

    ts_results = []

    for future_dt in future_hours:
        df_temp = project_latest_sequence_to_future(df, future_dt, scenario)

        X_temp = build_sequence_tensor(df_temp, sensor_order, TARGET_F)

        with torch.no_grad():
            logits_temp = model(X_temp, A)
            probs_temp = torch.softmax(logits_temp, dim=-1)

            pred_temp = logits_temp.argmax(dim=-1).squeeze(0).detach().cpu().numpy()
            conf_temp = probs_temp.max(dim=-1).values.squeeze(0).detach().cpu().numpy()

        sensor_idx = sensor_order.index(selected_sensor)
        plos_value = int(pred_temp[sensor_idx])
        count_value = int(round(plos_count_lookup.get(plos_value, 0)))

        ts_results.append({
            "datetime": future_dt,
            "hour": future_dt.hour,
            "PLOS": plos_value,
            "predicted_count": count_value,
            "confidence": float(conf_temp[sensor_idx]),
            "scenario": scenario,
            "sensor_id": selected_sensor,
        })

    ts_df = pd.DataFrame(ts_results)

    col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)

    col_ts1.metric("Selected Sensor", selected_sensor)
    col_ts2.metric("Avg 24h PLOS", round(ts_df["PLOS"].mean(), 2))
    col_ts3.metric("Max 24h PLOS", int(ts_df["PLOS"].max()))
    col_ts4.metric("Avg 24h Count", int(ts_df["predicted_count"].mean()))

    st.line_chart(
        ts_df.set_index("datetime")[["PLOS", "predicted_count"]]
    )

    st.dataframe(ts_df)

    st.download_button(
        "Download Selected Sensor 24h Profile",
        ts_df.to_csv(index=False),
        f"{selected_sensor}_24h_future_plos_count_profile.csv",
        "text/csv",
    )


# ============================================================
# SUMMARY
# ============================================================

st.subheader("📊 Prediction Summary")

c1,  c3, c5 = st.columns(3)

c1.metric(
    "Avg Sensed PLOS",
    round(sensor_map_all["PLOS"].mean(), 2)
)




c3.metric(
    "Avg Unsensed PLOS",
    round(unsensed_pred_all["PLOS"].mean(), 2)
)




c5.metric(
    "Avg Predicted Pedestrian Count",
    int(sensor_map_all["predicted_count"].mean())
)


st.subheader("📈 PLOS Distribution")

dist_sensed = sensor_map_all["PLOS"].value_counts().sort_index()
dist_unsensed = unsensed_pred_all["PLOS"].value_counts().sort_index()

dist = pd.DataFrame({
    "Sensed": dist_sensed,
    "Unsensed": dist_unsensed,
}).fillna(0).astype(int)

st.bar_chart(dist)


st.subheader("📈 Pedestrian Count Distribution")

count_dist = pd.DataFrame({
    "Sensed predicted count": sensor_map_all["predicted_count"],
    "Unsensed predicted count": unsensed_pred_all["predicted_count"].head(len(sensor_map_all)).reset_index(drop=True),
})

st.bar_chart(count_dist)


# ============================================================
# HIGH-RISK TABLES
# ============================================================

st.subheader("⚠️ High-Risk Locations")

tab1, tab2 = st.tabs(["Sensed Sensors", "Unsensed Locations"])

with tab1:
    sensed_risk = sensor_map_all[sensor_map_all["PLOS"] >= 4].copy()
    st.write(f"High-risk sensed sensors: {len(sensed_risk)}")

    st.dataframe(
        sensed_risk[
            [
                "sensor_id",
                "lat",
                "lon",
                "PLOS",
                "predicted_count",
                "confidence",
                "selected_datetime",
                "scenario",
            ]
        ]
        .sort_values(["PLOS", "predicted_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

with tab2:
    unsensed_risk = unsensed_pred_all[unsensed_pred_all["PLOS"] >= 4].copy()
    st.write(f"High-risk unsensed locations: {len(unsensed_risk)}")

    st.dataframe(
        unsensed_risk[
            [
                "unsensed_id",
                "lat",
                "lon",
                "PLOS",
                "predicted_count",
                "PLOS_score",
                "nearest_distance_m",
                "confidence",
                "selected_datetime",
                "scenario",
            ]
        ]
        .sort_values(["PLOS", "predicted_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


# ============================================================
# METHOD NOTE
# ============================================================

with st.expander("Method and interpretation note"):
    st.write(
        """
        This interface performs scenario-based future simulation. The ST-Graph Transformer
        predicts six-class PLOS for sensed sensor nodes using the latest 24-hour sequence
        projected into the selected future temporal context.

        Pedestrian count is reported as an auxiliary demand estimate by mapping each predicted
        PLOS class to the median observed pedestrian count associated with that class in the
        training data. This provides interpretable pedestrian volume estimates while keeping
        the ordinal PLOS model as the main prediction task.

        Unsensed locations are estimated using graph propagation from nearby sensed sensors
        through inverse-distance weighting.

        The selected-sensor time-series panel repeatedly projects the 24-hour input window
        into each hour of the selected future day, producing a 24-hour PLOS and pedestrian
        count profile for local planning inspection.

        The future date selection should be interpreted as a digital-twin scenario simulation,
        not as direct observation of future ground truth.
        """
    )

    st.write(f"Selected datetime: {selected_datetime}")
    st.write(f"Scenario: {scenario}")
    st.write(f"Selected sensor: {selected_sensor}")
    st.write(f"Model input feature count: {TARGET_F}")
    st.write(f"Number of sensed nodes: {len(sensor_order)}")
    st.write(f"Adjacency shape: {tuple(A.shape)}")
    st.write("PLOS → count lookup:", plos_count_lookup)


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader("⬇️ Download Results")

col_a, col_b = st.columns(2)

with col_a:
    st.download_button(
        "Download Sensed Predictions",
        sensor_map_all.to_csv(index=False),
        "tel_aviv_sensed_future_plos_count_predictions.csv",
        "text/csv",
    )

with col_b:
    st.download_button(
        "Download Unsensed Predictions",
        unsensed_pred_all.to_csv(index=False),
        "tel_aviv_unsensed_future_plos_count_predictions.csv",
        "text/csv",
    )
