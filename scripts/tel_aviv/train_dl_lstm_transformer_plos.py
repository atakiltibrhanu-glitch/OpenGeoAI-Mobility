import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = r"D:\Morphology_Aware\data\processed\Tel_Aviv\tel_aviv_multimodal_2023_REALISTIC_WITH_TARGET_AND_LAG.csv"

SEQ_LEN = 24
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3

LOS_LABELS = 6

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# remove lag for fair comparison
df = df.drop(columns=["lag1","lag_24","lag_168","lag1_x_peak"], errors="ignore")

# ============================================================
# CREATE PLOS
# ============================================================

thresholds = np.percentile(df["count"], [20,40,60,80,90])

def to_plos(x):
    if x <= thresholds[0]: return 0
    elif x <= thresholds[1]: return 1
    elif x <= thresholds[2]: return 2
    elif x <= thresholds[3]: return 3
    elif x <= thresholds[4]: return 4
    else: return 5

df["PLOS"] = df["count"].apply(to_plos)

# ============================================================
# SPLIT
# ============================================================

train_df = df[df["timestamp"] < "2023-10-01"]
test_df  = df[df["timestamp"] >= "2023-10-01"]

print("Train:", train_df.shape)
print("Test :", test_df.shape)

# ============================================================
# FEATURES
# ============================================================

drop_cols = ["count","PLOS","timestamp","sensor_name","city","country"]
features = [c for c in df.columns if c not in drop_cols]

X_train = pd.get_dummies(train_df[features], dummy_na=True)
X_test  = pd.get_dummies(test_df[features], dummy_na=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train_df["PLOS"].values
y_test  = test_df["PLOS"].values

X_train = X_train.values.astype(np.float32)
X_test  = X_test.values.astype(np.float32)

# ============================================================
# SEQUENCE DATASET
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.seq_len],
            self.y[idx+self.seq_len]
        )

train_ds = SeqDataset(X_train, y_train, SEQ_LEN)
test_ds  = SeqDataset(X_test, y_test, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

INPUT_DIM = X_train.shape[1]

# ============================================================
# MODEL 1: LSTM
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, LOS_LABELS)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ============================================================
# MODEL 2: TRANSFORMER
# ============================================================

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, LOS_LABELS)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1])

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(model):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            out = model(Xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    return model

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb)
            pred = out.argmax(dim=1).cpu().numpy()

            preds.extend(pred)
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    qwk = cohen_kappa_score(trues, preds, weights="quadratic")
    f1 = f1_score(trues, preds, average="weighted")

    print("\nRESULTS")
    print("Accuracy:", acc)
    print("QWK     :", qwk)
    print("F1      :", f1)

# ============================================================
# RUN
# ============================================================

print("\nTraining LSTM...")
lstm = train_model(LSTMModel(INPUT_DIM))
evaluate(lstm)

print("\nTraining Transformer...")
trans = train_model(TransformerModel(INPUT_DIM))
evaluate(trans)