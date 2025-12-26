import json
import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from dataset import FeatureDescription, TimeSeriesDataset
from model import TemporalFusionTransformer, quantile_loss

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return yaml.safe_load(raw), raw


def single_quantile_loss(y_true, y_pred, q):
    errors = y_true - y_pred
    loss = torch.maximum(q * errors, (q - 1) * errors)
    return loss.mean()


def find_quantile_index(quantiles, target):
    for i, q in enumerate(quantiles):
        if abs(float(q) - target) < 1e-6:
            return i
    return None


config, config_raw = load_config("config.yaml")

training_cfg = config.get("training", {})
log_root = training_cfg.get("log_dir", "runs")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(log_root, timestamp)
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(config_raw)

writer = SummaryWriter(log_dir=run_dir)

# Create feature description for electricity dataset
feature_cfg = config["feature_description"]
feature_description = FeatureDescription(
    id=feature_cfg["id"],
    time=feature_cfg["time"],
    target=feature_cfg["target"],
    known_continuous=feature_cfg.get("known_continuous", []),
    known_categorical=feature_cfg.get("known_categorical", []),
    static_categorical=feature_cfg.get("static_categorical", []),
    static_continuous=feature_cfg.get("static_continuous", []),
    observed_continuous=feature_cfg.get("observed_continuous", []),
    observed_categorical=feature_cfg.get("observed_categorical", []),
)

# Load dataset
dataset_cfg = config["dataset"]
df = pd.read_csv(dataset_cfg["path"])

# Split into train, val, test
valid_boundary = dataset_cfg["valid_boundary"]
test_boundary = dataset_cfg["test_boundary"]

df_train = df[df["days_from_start"] < valid_boundary]
df_val = df[(df["days_from_start"] >= valid_boundary - 7) & (df["days_from_start"] < test_boundary)]
df_test = df[df["days_from_start"] >= test_boundary - 7]

# Create datasets
model_cfg = config["model"]
train_dataset = TimeSeriesDataset(
    df=df_train,
    feature_description=feature_description,
    encoder_length=model_cfg["encoder_length"],
    decoder_length=model_cfg["decoder_length"],
)
# Get categorical encoder and scalers from training set
categorical_encoder = train_dataset.categorical_encoder
real_scalers, target_scalers = TimeSeriesDataset.get_scalers(train_dataset)

val_dataset = TimeSeriesDataset(
    df=df_val,
    feature_description=feature_description,
    encoder_length=model_cfg["encoder_length"],
    decoder_length=model_cfg["decoder_length"],
    categorical_encoder=categorical_encoder,
)

test_dataset = TimeSeriesDataset(
    df=df_test,
    feature_description=feature_description,
    encoder_length=model_cfg["encoder_length"],
    decoder_length=model_cfg["decoder_length"],
    categorical_encoder=categorical_encoder,
)

# Apply scalers
train_dataset.apply_scalers(real_scalers, target_scalers)
val_dataset.apply_scalers(real_scalers, target_scalers)
test_dataset.apply_scalers(real_scalers, target_scalers)

dataloader_cfg = config.get("dataloader", {})
batch_size = dataloader_cfg.get("batch_size", 64)
num_workers = dataloader_cfg.get("num_workers", 0)
pin_memory = dataloader_cfg.get("pin_memory", False)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
)

device_cfg = training_cfg.get("device", "auto")
if device_cfg == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_cfg)

# Create params
params = {
    "encoder_length": train_dataset.enc_len,
    "decoder_length": train_dataset.dec_len,
    "time_steps": train_dataset.time_steps,
    "feature_description": feature_description,
    "embed_per_cat": train_dataset.get_embedding_per_cat(),
    "d_model": model_cfg["d_model"],
    "dropout": model_cfg["dropout"],
    "n_head": model_cfg["n_head"],
    "quantiles": model_cfg["quantiles"],
}

model = TemporalFusionTransformer(params=params).to(device)

opt = optim.Adam(model.parameters(), lr=training_cfg.get("learning_rate", 1e-3))
grad_clip = training_cfg.get("grad_clip", 1.0)

target_quantiles = [0.5, 0.9]
quantile_indices = {q: find_quantile_index(model.quantiles, q) for q in target_quantiles}
missing_quantiles = [q for q, idx in quantile_indices.items() if idx is None]
if missing_quantiles:
    print(f"Warning: missing quantiles in model.quantiles: {missing_quantiles}")


@torch.no_grad()
def run_epoch(model, loader, quantiles, quantile_indices):
    model.eval()
    tot_loss, n = 0.0, 0
    q_totals = {q: 0.0 for q, idx in quantile_indices.items() if idx is not None}
    for batch in loader:
        preds = model(batch)  # [B, Td, Q]
        loss = quantile_loss(batch["target"], preds, quantiles)
        bs = batch["target"].shape[0]
        tot_loss += loss.item() * bs
        for q, idx in quantile_indices.items():
            if idx is None:
                continue
            q_loss = single_quantile_loss(batch["target"], preds[..., idx:idx + 1], q)
            q_totals[q] += q_loss.item() * bs
        n += bs
    avg_loss = tot_loss / max(n, 1)
    q_avgs = {q: total / max(n, 1) for q, total in q_totals.items()}
    return avg_loss, q_avgs


epochs = training_cfg.get("epochs", 1)
train_hist, val_hist = [], []
history = []

for epoch in range(1, epochs + 1):
    model.train()
    running, nseen = 0.0, 0
    q_running = {q: 0.0 for q, idx in quantile_indices.items() if idx is not None}
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
        opt.zero_grad()
        preds = model(batch)
        loss = quantile_loss(batch["target"], preds, model.quantiles)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        bs = batch["target"].shape[0]
        running += loss.item() * bs
        for q, idx in quantile_indices.items():
            if idx is None:
                continue
            q_loss = single_quantile_loss(batch["target"], preds[..., idx:idx + 1], q)
            q_running[q] += q_loss.item() * bs
        nseen += bs

    train_loss = running / max(nseen, 1)
    train_q = {q: total / max(nseen, 1) for q, total in q_running.items()}
    val_loss, val_q = run_epoch(model, val_loader, model.quantiles, quantile_indices)

    train_hist.append(train_loss)
    val_hist.append(val_loss)
    history_entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
    for q in target_quantiles:
        if q in train_q:
            history_entry[f"train_p{int(q * 100)}"] = train_q[q]
        if q in val_q:
            history_entry[f"val_p{int(q * 100)}"] = val_q[q]
    history.append(history_entry)

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)
    if 0.5 in train_q:
        writer.add_scalar("loss/train_p50", train_q[0.5], epoch)
    if 0.5 in val_q:
        writer.add_scalar("loss/val_p50", val_q[0.5], epoch)
    if 0.9 in train_q:
        writer.add_scalar("loss/train_p90", train_q[0.9], epoch)
    if 0.9 in val_q:
        writer.add_scalar("loss/val_p90", val_q[0.9], epoch)
    print(f"Epoch {epoch:02d}: train={train_loss:.5f}  val={val_loss:.5f}")

with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

writer.close()
