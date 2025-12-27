import yaml
import torch


def load_config(path, return_raw=False):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    config = yaml.safe_load(raw)
    if return_raw:
        return config, raw
    return config


def single_quantile_loss(y_true, y_pred, q):
    errors = y_true - y_pred
    loss = torch.maximum(q * errors, (q - 1) * errors)
    return loss.mean()


def find_quantile_index(quantiles, target):
    """Finds the index of a specific quantile in the list."""
    for i, q in enumerate(quantiles):
        if abs(float(q) - target) < 1e-6:
            return i
    return None


def unscale_per_id(preds, targets, ids, target_scalers):
    """Unscales predictions and targets using the per-ID scalers."""
    # Move to CPU for plotting/metrics if needed, or keep on device
    device = preds.device
    
    # Retrieve means and scales for the current batch of IDs
    scales = []
    means = []
    for id_val in ids:
        # id_val might be a tensor or string depending on collate
        id_str = str(id_val.item()) if isinstance(id_val, torch.Tensor) else str(id_val)
        scaler = target_scalers[id_str]
        scales.append(scaler.scale_[0])
        means.append(scaler.mean_[0])
    
    scales = torch.tensor(scales, device=device, dtype=preds.dtype).view(-1, 1, 1)
    means = torch.tensor(means, device=device, dtype=preds.dtype).view(-1, 1, 1)
    
    # Apply inverse transform: x * scale + mean
    preds_unscaled = preds * scales + means
    targets_unscaled = targets * scales + means
    
    return preds_unscaled, targets_unscaled


def update_qrisk_totals(preds, targets, ids, quantile_indices, target_scalers_by_id):
    device = preds.device
    
    scales = []
    means = []
    for id_val in ids:
        scaler = target_scalers_by_id[str(id_val)]
        scales.append(scaler.scale_[0])
        means.append(scaler.mean_[0])
    
    scales_tensor = torch.tensor(scales, device=device, dtype=preds.dtype).view(-1, 1, 1)
    means_tensor = torch.tensor(means, device=device, dtype=preds.dtype).view(-1, 1, 1)
    targets_unscaled = targets * scales_tensor + means_tensor
    preds_unscaled = preds * scales_tensor + means_tensor
    target_abs_total = targets_unscaled.abs().sum().item()
    
    qloss_totals = {}
    for q, idx in quantile_indices.items():
        if idx is not None:
            pred_q = preds_unscaled[..., idx:idx+1]
            errors = targets_unscaled - pred_q
            loss = torch.maximum(q * errors, (q - 1) * errors)
            qloss_totals[q] = loss.sum().item()

    return qloss_totals, target_abs_total
