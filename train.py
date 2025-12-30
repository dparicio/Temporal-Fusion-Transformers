import json
import os
from datetime import datetime
import wandb

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from dataset import FeatureDescription, TimeSeriesDataset
from model import TemporalFusionTransformer, quantile_loss
from utils import load_config, find_quantile_index, update_qrisk_totals, single_quantile_loss

config, config_raw = load_config("config.yaml", return_raw=True)
wandb_cfg = config.get("wandb", {})
wandb_enabled = bool(wandb_cfg.get("enabled", False))


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch

training_cfg = config.get("training", {})
log_root = training_cfg.get("log_dir", "runs")
resume_from = training_cfg.get("resume_from")
if resume_from:
    resume_from = os.path.expanduser(resume_from)
    if os.path.exists(resume_from):
        run_dir = os.path.abspath(os.path.join(os.path.dirname(resume_from), os.pardir))
    else:
        print(f"Resume checkpoint not found: {resume_from}. Starting new run.")
        resume_from = None
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(log_root, timestamp)
else:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(log_root, timestamp)
os.makedirs(run_dir, exist_ok=True)

config_path = os.path.join(run_dir, "config.yaml")
if not os.path.exists(config_path):
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_raw)

writer = SummaryWriter(log_dir=run_dir)
wandb_run = None
if wandb_enabled:
    wandb_init_kwargs = {
        "project": wandb_cfg.get("project", "tft"),
        "config": config,
        "mode": wandb_cfg.get("mode", "online"),
        "dir": run_dir,
    }
    if wandb_cfg.get("entity"):
        wandb_init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("name"):
        wandb_init_kwargs["name"] = wandb_cfg["name"]
    if wandb_cfg.get("tags"):
        wandb_init_kwargs["tags"] = wandb_cfg["tags"]
    if wandb_cfg.get("notes"):
        wandb_init_kwargs["notes"] = wandb_cfg["notes"]
    wandb_run = wandb.init(**wandb_init_kwargs)

ckpt_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

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
if isinstance(feature_description.target, (list, tuple)):
    target_cols = list(feature_description.target)
else:
    target_cols = [feature_description.target]
missing_targets = [t for t in target_cols if t not in feature_description.observed_continuous]
if missing_targets:
    feature_description.observed_continuous = feature_description.observed_continuous + missing_targets

# Load dataset
dataset_cfg = config["dataset"]
df = pd.read_csv(dataset_cfg["path"])

# Model config 
model_cfg = config["model"]

# Split into train, val, test using time column boundaries
time_col = feature_description.time
valid_boundary = dataset_cfg["valid_boundary"]
test_boundary = dataset_cfg["test_boundary"]
step_seconds = dataset_cfg.get("step_seconds")
if step_seconds is None:
    time_values = pd.to_numeric(df[time_col], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(time_values) < 2:
        raise ValueError("Cannot infer step_seconds from time column; set dataset.step_seconds in config.")
    step_seconds = int(time_values.diff().median())
overlap = model_cfg["encoder_length"] * step_seconds

df_train = df[df[time_col] < valid_boundary]
df_val = df[(df[time_col] >= valid_boundary - overlap) & (df[time_col] < test_boundary)]
df_test = df[df[time_col] >= test_boundary - overlap]

id_col = feature_description.id
train_ids = set(df_train[id_col].unique())
df_val = df_val[df_val[id_col].isin(train_ids)]
df_test = df_test[df_test[id_col].isin(train_ids)]

# Create datasets
train_dataset = TimeSeriesDataset(
    df=df_train,
    feature_description=feature_description,
    encoder_length=model_cfg["encoder_length"],
    decoder_length=model_cfg["decoder_length"],
)
# Get categorical encoder and scalers from training set
categorical_encoder = train_dataset.categorical_encoder
real_scalers, target_scalers = TimeSeriesDataset.get_scalers(train_dataset)
target_scalers_by_id = {str(k): v for k, v in target_scalers.items()}

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
default_num_workers = training_cfg.get("multiprocessing_workers", 0)
num_workers = dataloader_cfg.get("num_workers", default_num_workers)
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
steps_per_epoch = max(len(train_loader), 1)
decay_per_epoch = training_cfg.get("decay_per_epoch", 0.5)
scheduler = optim.lr_scheduler.LambdaLR(
    opt,
    lr_lambda=lambda step: decay_per_epoch ** (step / steps_per_epoch),
)
grad_clip = training_cfg.get("grad_clip", 1.0)
log_every = training_cfg.get("log_every", 50)
log_val_every = training_cfg.get("log_val_every", log_every)
log_val_enabled = training_cfg.get("log_val", True)
early_stopping_patience = training_cfg.get("early_stopping_patience", 5)
early_stopping_min_delta = training_cfg.get("early_stopping_min_delta", 1e-4)
if early_stopping_patience is not None and early_stopping_patience <= 0:
    early_stopping_patience = None
if early_stopping_min_delta is None:
    early_stopping_min_delta = 0.0

start_epoch = 1
best_val_loss = float("inf")
epochs_no_improve = 0
global_step = 0
if resume_from:
    checkpoint = torch.load(resume_from, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    if "optimizer_state" in checkpoint:
        opt.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resumed from {resume_from} at epoch {start_epoch}")

target_quantiles = [0.5, 0.9]
quantile_indices = {q: find_quantile_index(model.quantiles, q) for q in target_quantiles}
missing_quantiles = [q for q, idx in quantile_indices.items() if idx is None]
if missing_quantiles:
    print(f"Warning: missing quantiles in model.quantiles: {missing_quantiles}")


@torch.no_grad()
def run_epoch(model, loader, quantiles, quantile_indices, target_scalers_by_id):
    model.eval()
    tot_loss, n = 0.0, 0
    q_totals = {q: 0.0 for q, idx in quantile_indices.items() if idx is not None}
    qloss_totals = {q: 0.0 for q, idx in quantile_indices.items() if idx is not None}
    target_abs_total = 0.0
    device = next(model.parameters()).device
    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)  # [B, Td, Q]
        targets = batch["target"].to(preds.device)
        loss = quantile_loss(targets, preds, quantiles)
        bs = batch["target"].shape[0]
        tot_loss += loss.item() * bs
        for q, idx in quantile_indices.items():
            if idx is None:
                continue
            q_loss = single_quantile_loss(targets, preds[..., idx:idx + 1], q)
            q_totals[q] += q_loss.item() * bs
        batch_qloss_totals, batch_target_abs_total = update_qrisk_totals(
            preds=preds,
            targets=targets,
            ids=batch["id"],
            quantile_indices=quantile_indices,
            target_scalers_by_id=target_scalers_by_id,
        )
        for q in qloss_totals:
            qloss_totals[q] += batch_qloss_totals[q]
        target_abs_total += batch_target_abs_total
        n += bs
    avg_loss = tot_loss / max(n, 1)
    q_avgs = {q: total / max(n, 1) for q, total in q_totals.items()}
    if target_abs_total > 0.0:
        q_risks = {q: 2.0 * total / target_abs_total for q, total in qloss_totals.items()}
    else:
        q_risks = {q: float("nan") for q in qloss_totals}
    return avg_loss, q_avgs, q_risks


epochs = training_cfg.get("epochs", 1)
train_hist, val_hist = [], []
history = []

for epoch in range(start_epoch, epochs + 1):
    model.train()
    running, nseen = 0.0, 0
    q_running = {q: 0.0 for q, idx in quantile_indices.items() if idx is not None}
    window_running, window_nseen = 0.0, 0
    window_q_running = {q: 0.0 for q, idx in quantile_indices.items() if idx is not None}
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)):
        batch = move_to_device(batch, device)
        opt.zero_grad()
        preds = model(batch)
        loss = quantile_loss(batch["target"], preds, model.quantiles)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        bs = batch["target"].shape[0]
        running += loss.item() * bs
        window_running += loss.item() * bs
        for q, idx in quantile_indices.items():
            if idx is None:
                continue
            q_loss = single_quantile_loss(batch["target"], preds[..., idx:idx + 1], q)
            q_running[q] += q_loss.item() * bs
            window_q_running[q] += q_loss.item() * bs
        nseen += bs
        window_nseen += bs
        global_step += 1
        scheduler.step(global_step)

        if log_every and (batch_idx + 1) % log_every == 0:
            window_loss = window_running / max(window_nseen, 1)
            writer.add_scalar("loss/train_step", window_loss, global_step)
            for q, total in window_q_running.items():
                writer.add_scalar(f"loss/train_p{int(q * 100)}_step", total / max(window_nseen, 1), global_step)
            if wandb_run is not None:
                step_metrics = {
                    "loss/train_step": window_loss,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                for q, total in window_q_running.items():
                    step_metrics[f"loss/train_p{int(q * 100)}_step"] = total / max(window_nseen, 1)
                wandb_run.log(step_metrics, step=global_step)
            window_running, window_nseen = 0.0, 0
            window_q_running = {q: 0.0 for q, idx in quantile_indices.items() if idx is not None}

        if log_val_enabled and log_val_every and (batch_idx + 1) % log_val_every == 0:
            val_loss_step, val_q_step, val_qrisk_step = run_epoch(
                model,
                val_loader,
                model.quantiles,
                quantile_indices,
                target_scalers_by_id,
            )
            writer.add_scalar("loss/val_step", val_loss_step, global_step)
            for q, val in val_q_step.items():
                writer.add_scalar(f"loss/val_p{int(q * 100)}_step", val, global_step)
            for q, val in val_qrisk_step.items():
                writer.add_scalar(f"qrisk/val_p{int(q * 100)}_step", val, global_step)
            if wandb_run is not None:
                val_step_metrics = {
                    "loss/val_step": val_loss_step,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                for q, val in val_q_step.items():
                    val_step_metrics[f"loss/val_p{int(q * 100)}_step"] = val
                for q, val in val_qrisk_step.items():
                    val_step_metrics[f"qrisk/val_p{int(q * 100)}_step"] = val
                wandb_run.log(val_step_metrics, step=global_step)
            model.train()

    train_loss = running / max(nseen, 1)
    train_q = {q: total / max(nseen, 1) for q, total in q_running.items()}
    val_loss, val_q, val_qrisk = run_epoch(
        model,
        val_loader,
        model.quantiles,
        quantile_indices,
        target_scalers_by_id,
    )

    train_hist.append(train_loss)
    val_hist.append(val_loss)
    history_entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
    for q in target_quantiles:
        if q in train_q:
            history_entry[f"train_p{int(q * 100)}"] = train_q[q]
        if q in val_q:
            history_entry[f"val_p{int(q * 100)}"] = val_q[q]
        if q in val_qrisk:
            history_entry[f"val_qrisk_p{int(q * 100)}"] = val_qrisk[q]
    history.append(history_entry)

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)
    writer.add_scalar("loss/train_p50", train_q[0.5], epoch)
    writer.add_scalar("loss/val_p50", val_q[0.5], epoch)
    writer.add_scalar("loss/train_p90", train_q[0.9], epoch)
    writer.add_scalar("loss/val_p90", val_q[0.9], epoch)
    writer.add_scalar("qrisk/val_p50", val_qrisk[0.5], epoch)
    writer.add_scalar("qrisk/val_p90", val_qrisk[0.9], epoch)
    if wandb_run is not None:
        epoch_metrics = {
            "loss/train": train_loss,
            "loss/val": val_loss,
            "loss/train_p50": train_q[0.5],
            "loss/val_p50": val_q[0.5],
            "loss/train_p90": train_q[0.9],
            "loss/val_p90": val_q[0.9],
            "qrisk/val_p50": val_qrisk[0.5],
            "qrisk/val_p90": val_qrisk[0.9],
            "lr": opt.param_groups[0]["lr"],
            "epoch": epoch,
            "global_step": global_step,
        }
        wandb_run.log(epoch_metrics, step=global_step)
    is_best = val_loss < (best_val_loss - early_stopping_min_delta)
    if is_best:
        best_val_loss = val_loss
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "best_val_loss": best_val_loss,
    }
    checkpoint["scheduler_state"] = scheduler.state_dict()
    torch.save(checkpoint, last_ckpt_path)
    if is_best:
        torch.save(checkpoint, best_ckpt_path)
    val_qrisk_p50 = val_qrisk.get(0.5, float("nan"))
    val_qrisk_p90 = val_qrisk.get(0.9, float("nan"))
    print(
        f"Epoch {epoch:02d}: train={train_loss:.5f}  val={val_loss:.5f}  "
        f"qrisk_p50={val_qrisk_p50:.5f}  qrisk_p90={val_qrisk_p90:.5f}"
    )
    if early_stopping_patience is not None:
        if is_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch:02d} "
                    f"(no improvement in val loss for {early_stopping_patience} epochs)."
                )
                break

with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

writer.close()
if wandb_run is not None:
    wandb_run.finish()
