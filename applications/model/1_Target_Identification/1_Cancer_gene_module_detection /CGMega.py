#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGMega model/training module.

This file intentionally does not read raw data matrices. Raw data access is
restricted to cgmega_data.py. This file can only obtain a processed
CancerDataset via get_data(), then build loaders and train/test CGMega.

CLI exposes only paths and runtime flags. Hyperparameters are fixed below.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn.functional as F
from torch.nn import Dropout, MaxPool1d

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import dropout_adj

try:
    from transformers import get_linear_schedule_with_warmup
except Exception:
    get_linear_schedule_with_warmup = None



# =============================================================================
# 0. Data module dynamic loading
# =============================================================================


def load_cgmega_data_module(cgmega_data_path: str) -> ModuleType:
    """Load cgmega_data.py from a path explicitly provided by run.sh/CLI.

    The model/training file intentionally does not assume cgmega_data.py is in
    the same directory. This keeps the data-processing module location under
    external runtime control while still restricting raw data access to that
    module.
    """
    module_path = Path(cgmega_data_path).expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"cgmega_data.py does not exist: {module_path}")
    if module_path.name != "cgmega_data.py":
        raise ValueError(f"Expected a file named cgmega_data.py, got: {module_path}")

    spec = importlib.util.spec_from_file_location("cgmega_data", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Register before execution so pickle/module aliases inside cgmega_data.py work.
    sys.modules["cgmega_data"] = module
    spec.loader.exec_module(module)

    for required_name in ("CancerDataset", "get_data", "get_data_root"):
        if not hasattr(module, required_name):
            raise AttributeError(f"{module_path} is missing required symbol: {required_name}")

    return module


# =============================================================================
# 1. Default hyperparameters
# =============================================================================

# Core hyperparameters are intentionally stored inside this file.
# Values are copied from run.sh and are no longer controlled by CLI arguments.
CORE_HYPERPARAMETERS: Dict[str, object] = {
    "num_epochs": 500,
    "batch_size": 256,
    "cv_folds": 10,
    "repeat": 1,
    "random_seed": 42,
    "hidden_channels": 128,
    "heads": 3,
    "drop_rate": 0.4,
    "ppi_attn_drop": 0.1,
    "lr": 0.001,
    "weight_decay": 0.05,
    "sample_neg": 0.5,
    "sample_pos": 1.0,
    "sample_rate": 1.0,
}

# Data and graph hyperparameters are intentionally stored inside this file.
# Values are copied from run.sh and are no longer controlled by CLI arguments.
DATA_GRAPH_HYPERPARAMETERS: Dict[str, object] = {
    "ppi": "CPDB",
    "graph": "ppi",
    "hic": True,
    "hic_reduce": "svd",
    "hic_reduce_dim": 5,
    "hic_norm": "log",
    "hic_type": "ice",
    "hic_drop_rate": 0.0,
    "ppi_drop_rate": 0.0,
    "resolution": "10KB",
    "stable": True,
}

DEFAULT_CONFIG: Dict[str, object] = {
    # Runtime/path values are filled by CLI in build_config().
    "data_dir": None,
    "data_root": None,
    "ppi_dir": None,
    "cgmega_data_path": None,
    "joint": False,
    "load_data": True,
    "log_dir": None,
    "log_name": None,
    "logfile": None,
    "model": "CGMega",
    "neighbors": [-1, -1],
    "out_dir": None,
    "reverse": False,
    "device": "cpu",
    "fold": 0,
    **CORE_HYPERPARAMETERS,
    **DATA_GRAPH_HYPERPARAMETERS,
}


# Training/model constants.
HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2


# =============================================================================
# 2. Non-data utility functions
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config(filename: str, configs: Dict[str, object]) -> None:
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(filename, "a") as f:
        for key, val in configs.items():
            print(key, ":", val, file=f, flush=True)


# =============================================================================
# 4. CGMega model only
# =============================================================================

@dataclass(frozen=True)
class ModelInputSpec:
    """Only dataset-derived metadata visible to the model architecture."""

    in_channels: int
    edge_dim: int


def infer_model_input_spec(dataset: Any) -> ModelInputSpec:
    """Extract the minimal shape metadata needed to initialize CGMega."""
    data = dataset[0]
    return ModelInputSpec(
        in_channels=int(dataset.num_node_features),
        edge_dim=int(data.edge_dim),
    )


class CGMega(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, residual,
                 devices_available):
        super(CGMega, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate
        self.convs = t.nn.ModuleList()
        self.residual = residual
        mid_channels = in_channels + hidden_channels if residual else hidden_channels

        self.convs.append(
            TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim,
                            concat=False, beta=True).to(self.devices_available))
        self.convs.append(TransformerConv(mid_channels, hidden_channels, heads=heads,
                                          dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True).to(
            self.devices_available))

        self.ln1 = LayerNorm(in_channels=mid_channels).to(self.devices_available)
        self.ln2 = LayerNorm(in_channels=hidden_channels *
                                         heads).to(self.devices_available)

        self.pool = MaxPool1d(2, 2)

        self.dropout = Dropout(drop_rate)
        self.lins = t.nn.ModuleList()
        self.lins.append(
            Linear(int(hidden_channels * heads / 2), HIDDEN_DIM, weight_initializer="kaiming_uniform").to(
                devices_available))
        self.lins.append(
            Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform").to(devices_available))

    def forward(self, data):
        data = data[0].to(self.devices_available)
        x = data.x
        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        res = x
        x = self.convs[0](x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.ln1(x)

        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        x = self.convs[1](x.to(self.devices_available), edge_index.to(
            self.devices_available), edge_attr.to(self.devices_available))
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)
        x = t.unsqueeze(x, 1)
        x = self.pool(x)
        x = t.squeeze(x)
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)

        return t.sigmoid(x)



# =============================================================================
# 5. Metrics
# =============================================================================

def safe_div(numerator, denominator) -> float:
    return float(numerator) / float(denominator) if float(denominator) != 0 else np.nan


def calculate_metrics(y_true, y_pred, y_score):
    """Calculate binary metrics used for validation/test evaluation.

    y_true must be binary labels. If y_pred is accidentally passed as
    continuous scores, it is converted by threshold 0.5. y_score remains
    continuous for AUROC/AUPRC.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    true_unique = np.unique(y_true)
    if not np.all(np.isin(true_unique, [0, 1, 0.0, 1.0])):
        raise ValueError(
            "y_true contains non-binary values. "
            f"First 20 y_true values: {y_true[:20]}. "
            "This usually means y_true and y_score were swapped."
        )

    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    pred_unique = np.unique(y_pred)
    if not np.all(np.isin(pred_unique, [0, 1, 0.0, 1.0])):
        y_pred = (y_pred.astype(float) > 0.5).astype(int)
    else:
        y_pred = y_pred.astype(int)

    acc = float(np.equal(y_true, y_pred).sum() / y_true.shape[0])
    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
    auprc = average_precision_score(y_true=y_true, y_score=y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan

    return acc, cf_matrix, auprc, f1, auc


def calculate_extended_binary_metrics(cf_matrix):
    """Return only the binary metrics kept in test_metrics.csv."""
    tn, fp, fn, tp = cf_matrix.ravel()
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div((tp * tn - fp * fn), mcc_den)

    return {
        "Precision": precision,
        "MCC": mcc,
        "Recall": recall,
        "Sensitivity": recall,
        "Specificity": specificity,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr,
    }

def get_test_metrics_csv_path(configs: Dict[str, object]) -> str:
    base, _ = os.path.splitext(str(configs["logfile"]))
    return base + "_test_metrics.csv"


def save_test_metrics_csv(configs: Dict[str, object], test_metric_rows: List[Dict[str, object]]) -> str:
    out_csv = get_test_metrics_csv_path(configs)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(test_metric_rows).to_csv(out_csv, index=False)
    return out_csv


def make_test_metric_row(
    configs,
    repeat,
    fold,
    split_name,
    y_true,
    acc,
    cf_matrix,
    auprc,
    f1,
    auc,
    checkpoint=None,
):
    """Build one compact test-metrics row.

    Non-metric metadata is preserved. Metric columns are restricted to:
    AUPRC, AUROC, ACC, F1, Precision, MCC, TN, FP, FN, TP,
    Recall, Sensitivity, Specificity, NPV, FPR, FNR.
    """
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = cf_matrix.ravel()
    extended = calculate_extended_binary_metrics(cf_matrix)

    return {
        # Non-metric metadata kept unchanged.
        "split": split_name,
        "repeat": repeat,
        "fold": fold,
        "model": configs.get("model", "CGMega"),
        "data_dir": configs.get("data_dir"),
        "ppi": configs.get("ppi"),
        "hic": configs.get("hic"),
        "graph": configs.get("graph"),
        "random_seed": configs.get("random_seed"),
        "threshold": 0.5,
        "n_test": int(y_true.shape[0]),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "checkpoint": checkpoint,

        # Requested metric columns only.
        "AUPRC": float(auprc),
        "AUROC": float(auc) if not pd.isna(auc) else np.nan,
        "ACC": float(acc),
        "F1": float(f1),
        "Precision": extended["Precision"],
        "MCC": extended["MCC"],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Recall": extended["Recall"],
        "Sensitivity": extended["Sensitivity"],
        "Specificity": extended["Specificity"],
        "NPV": extended["NPV"],
        "FPR": extended["FPR"],
        "FNR": extended["FNR"],
    }


# =============================================================================
# 6. Training and prediction
# =============================================================================

def drop_samples(
    dataset: Any,
    fold: int,
    sample_neg: float = 0.0,
    sample_pos: float = 1.0,
    num_samples: int = 0,
    random_seed: int = 42,
) -> List[int]:
    if sample_neg == 1 and sample_pos == 1:
        return []

    drop_neg = 1 - sample_neg
    drop_pos = 1 - sample_pos
    train_idx = dataset.get_idx_split(fold)["train"]

    neg_idx: List[int] = []
    pos_idx: List[int] = []
    for i in train_idx:
        idx = int(i.item())
        if dataset[0].y[idx][0]:
            neg_idx.append(idx)
        if dataset[0].y[idx][1]:
            pos_idx.append(idx)

    num_neg_samples = len(neg_idx)
    num_pos_samples = len(pos_idx)
    random.seed(random_seed)

    if num_samples:
        num_neg = int(num_samples * num_neg_samples / (num_neg_samples + num_pos_samples))
        num_pos = int(num_samples) - num_neg
        drop_neg_idx = random.sample(neg_idx, max(0, num_neg_samples - num_neg))
        drop_pos_idx = random.sample(pos_idx, max(0, num_pos_samples - num_pos))
    else:
        drop_neg_idx = random.sample(neg_idx, int(num_neg_samples * drop_neg))
        drop_pos_idx = random.sample(pos_idx, int(num_pos_samples * drop_pos))

    drop_idx = drop_neg_idx + drop_pos_idx
    print(
        f"Negatives: {num_neg_samples - len(drop_neg_idx)}, "
        f"Positives: {num_pos_samples - len(drop_pos_idx)}"
    )
    dataset[0].train_mask[drop_idx, fold] = False
    return drop_idx


def build_model(configs: Dict[str, object], input_spec: ModelInputSpec) -> CGMega:
    """Build CGMega from a restricted model input spec.

    The architecture receives only:
    - in_channels
    - edge_dim

    It does not receive the full dataset object or raw data paths.
    """
    return CGMega(
        in_channels=input_spec.in_channels,
        hidden_channels=int(configs["hidden_channels"]),
        heads=int(configs["heads"]),
        drop_rate=float(configs["drop_rate"]),
        attn_drop_rate=float(configs["ppi_attn_drop"]),
        edge_dim=input_spec.edge_dim,
        residual=True,
        devices_available=str(configs["device"]),
    )


def build_scheduler(optimizer, num_training_steps: int):
    num_training_steps = max(1, int(num_training_steps))
    warmup_steps = int(0.2 * num_training_steps)
    if get_linear_schedule_with_warmup is not None:
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_training_modules(configs: Dict[str, object], dataset: Any):
    fold = int(configs["fold"])
    print(
        "Drop ",
        1 - float(configs["sample_neg"]),
        "of negative train samples and ",
        1 - float(configs["sample_pos"]),
        "of positive train samples",
    )

    drop_idx = drop_samples(
        dataset,
        fold,
        sample_neg=float(configs["sample_neg"]),
        sample_pos=float(configs["sample_pos"]),
        random_seed=int(configs["random_seed"]),
    )

    if float(configs["sample_rate"]) < 1:
        print("Drop ", 1 - float(configs["sample_rate"]), "of train samples")
        drop_idx += drop_samples(
            dataset,
            fold,
            sample_neg=float(configs["sample_rate"]),
            sample_pos=float(configs["sample_rate"]),
            random_seed=int(configs["random_seed"]),
        )
    elif float(configs["sample_rate"]) > 1:
        print(f"Sample {configs['sample_rate']} train samples")
        drop_idx = drop_samples(
            dataset,
            fold,
            num_samples=int(configs["sample_rate"]),
            random_seed=int(configs["random_seed"]),
        )

    data = dataset[0]
    neighbors = configs["neighbors"]
    batch_size = int(configs["batch_size"])

    train_loader = NeighborLoader(
        data,
        num_neighbors=neighbors,
        batch_size=batch_size,
        directed=False,
        input_nodes=data.train_mask[:, fold],
        shuffle=True,
    )
    valid_loader = NeighborLoader(
        data,
        num_neighbors=neighbors,
        batch_size=batch_size,
        directed=False,
        input_nodes=data.valid_mask[:, fold],
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=neighbors,
        batch_size=batch_size,
        directed=False,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    input_spec = infer_model_input_spec(dataset)
    model = build_model(configs, input_spec)
    optimizer = t.optim.AdamW(
        [
            dict(params=model.convs.parameters(), weight_decay=float(configs["weight_decay"])),
            dict(params=model.lins.parameters(), weight_decay=float(configs["weight_decay"])),
        ],
        lr=float(configs["lr"]),
    )

    num_train_nodes = int(data.train_mask[:, fold].sum().item())
    num_training_steps = int(np.ceil(num_train_nodes / batch_size) * int(configs["num_epochs"]))
    scheduler = build_scheduler(optimizer, num_training_steps)

    return {
        "dataset": dataset,
        "model": model,
        "loss_func": torch.nn.BCELoss(),
        "train_loader_list": [train_loader],
        "valid_loader_list": [valid_loader],
        "test_loader_list": [test_loader],
        "optimizer": optimizer,
        "scheduler": scheduler,
        "drop_idx": drop_idx,
    }


def run_loader(model, loader_list, device: str, loss_func=None, return_loss: bool = False):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    y_index = []
    total_loss = 0.0
    steps = 0

    for data_tuple in zip(*loader_list):
        size = data_tuple[0].batch_size
        with torch.no_grad():
            out = model(data_tuple)[:size].view(-1)

        true_lab = data_tuple[0].y[:size][:, 1].to(device).float()
        pred_lab = torch.zeros(size, device=device)
        pred_lab[out > 0.5] = 1

        y_true.extend(true_lab.detach().cpu().numpy().tolist())
        y_pred.extend(pred_lab.detach().cpu().numpy().tolist())
        y_score.extend(out.detach().cpu().numpy().tolist())
        y_index.extend(data_tuple[0].pos[:size].detach().cpu().numpy().tolist())

        if return_loss and loss_func is not None:
            total_loss += loss_func(out, true_lab).item()
            steps += 1

    if return_loss:
        return (
            np.asarray(y_true),
            np.asarray(y_pred),
            np.asarray(y_score),
            np.asarray(y_index),
            total_loss / max(1, steps),
        )
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score), np.asarray(y_index)


def train_one_epoch(model, train_loader_list, optimizer, device: str, scheduler=None, loss_func=None) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for data_tuple in zip(*train_loader_list):
        optimizer.zero_grad()
        size = data_tuple[0].batch_size
        out = model(data_tuple)[:size].view(-1)
        true_lab = data_tuple[0].y[:size][:, 1].to(device).float()
        loss = loss_func(out, true_lab)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


def train_model(modules, configs, log_name: str, fold: int, head_info: bool = False):
    logfile = str(configs["logfile"])
    device = str(configs["device"])
    dataset = modules["dataset"]
    model = modules["model"]
    optimizer = modules["optimizer"]
    scheduler = modules["scheduler"]
    loss_func = modules["loss_func"]

    data = dataset[0]

    if head_info:
        print_config(logfile, configs)
        with open(logfile, "a") as f:
            print(
                "Model: CGMega\nTrain/Valid/Test:",
                int(data.train_mask[:, fold].sum()),
                int(data.valid_mask[:, fold].sum()),
                int(data.test_mask.sum()),
                file=f,
                flush=True,
            )
            print(model, file=f, flush=True)

    print("Start Training")

    best_auprc = -np.inf
    best_auc = np.nan
    best_acc = np.nan
    best_f1 = np.nan
    best_tp = 0
    best_epoch = -1
    best_checkpoint = None
    trigger_times = 0
    patience = max(1, int(configs["num_epochs"]) // 5)
    start_check_epoch = max(0, int(configs["num_epochs"]) // 10)

    for epoch in range(int(configs["num_epochs"])):
        train_loss = train_one_epoch(
            model,
            modules["train_loader_list"],
            optimizer,
            device,
            scheduler=scheduler,
            loss_func=loss_func,
        )

        y_true_train, y_pred_train, y_score_train, _ = run_loader(
            model, modules["train_loader_list"], device
        )
        train_acc, _, train_auprc, _, train_auc = calculate_metrics(
            y_true_train, y_pred_train, y_score_train
        )

        y_true, y_pred, y_score, _, valid_loss = run_loader(
            model,
            modules["valid_loader_list"],
            device,
            loss_func=loss_func,
            return_loss=True,
        )
        acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, "
                f"Acc: {acc:.4f}, AUPRC: {auprc:.4f}, TP: {cf_matrix[1, 1]}, "
                f"F1: {f1:.4f}, AUROC: {auc:.4f}, "
                f"Train AUPRC: {train_auprc:.4f}, Train AUROC: {train_auc:.4f}, Train ACC: {train_acc:.4f}"
            )

        if epoch >= start_check_epoch:
            if auprc < best_auprc:
                trigger_times += 1
                if trigger_times == patience:
                    print("Early Stopping")
                    break
            else:
                trigger_times = 0
                best_auprc = auprc
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_tp = int(cf_matrix[1, 1])
                best_epoch = epoch
                best_checkpoint = {
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                }

    if best_checkpoint is None:
        best_checkpoint = {
            "state_dict": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        }

    model_dir = Path(str(configs["out_dir"])) / log_name
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / f"{fold}_{best_auprc:.4f}_{best_auc:.4f}_{best_tp}.pkl"
    t.save(best_checkpoint, ckpt_path)

    with open(logfile, "a") as f:
        print(
            "epoch {}: AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}".format(
                best_epoch, best_auprc, best_auc, best_acc, best_f1, best_tp
            ),
            file=f,
            flush=True,
        )

    dataset[0].train_mask[modules["drop_idx"], fold] = True
    return best_auprc, best_auc, best_acc, best_f1, best_tp, str(ckpt_path)


def predict(model, loader_list, configs: Dict[str, object], ckpt: str):
    device = str(configs["device"])
    print(f"Loading model from {ckpt} ......")
    checkpoint = t.load(ckpt, map_location=model.devices_available)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true, y_pred, y_score, y_index = run_loader(model, loader_list, device)

    # Keep a single, explicit return order across the whole file.
    return (
        np.asarray(y_true).reshape(-1).astype(int),
        np.asarray(y_pred).reshape(-1).astype(int),
        np.asarray(y_score).reshape(-1).astype(float),
        np.asarray(y_index).reshape(-1),
    )


def pred_to_df(i: int, result, y_index, y_true, y_score):
    if i == 0:
        mid = np.array([y_index, y_true, y_score]).T
        return pd.DataFrame(data=mid, columns=["gene_index", "Label", f"score_{i}"])
    mid = pd.DataFrame(data=np.array([y_index, y_score]).T, columns=["gene_index", f"score_{i}"])
    return result.merge(mid)


def score_avg_performance(train_result: pd.DataFrame, score_col: List[str], logfile: str) -> None:
    folds = len(score_col)
    y_true = train_result["Label"].to_numpy()
    y_score = train_result["avg_score"].to_numpy()
    y_pred = train_result["pred_label"].to_numpy()
    acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)
    tp = cf_matrix[1, 1]
    with open(logfile, "a") as f:
        print(
            f"{folds}-folds AUPRC:{auprc:.4f}, AUROC:{auc:.4f}, "
            f"ACC:{acc:.4f}, F1:{f1:.4f}, TP:{tp:.1f}",
            file=f,
            flush=True,
        )


def cv_train(dataset: Any, configs: Dict[str, object], cross_validation: bool = True):
    log_name = str(configs["log_name"])
    num_folds = int(configs["cv_folds"])
    sum_auprc, sum_auc, sum_acc, sum_f1, sum_tp = [], [], [], [], []
    test_metric_rows: List[Dict[str, object]] = []

    for repeat_idx in range(int(configs["repeat"])):
        head_info = True if repeat_idx == 0 else False

        train_result = None
        fold_range = range(num_folds) if cross_validation else [int(configs.get("fold", 0))]

        for fold in fold_range:
            configs["fold"] = int(fold)
            modules = get_training_modules(configs, dataset)

            valid_auprc, valid_auc, valid_acc, valid_f1, valid_tp, ckpt = train_model(
                modules,
                configs,
                log_name,
                int(fold),
                head_info=head_info,
            )
            head_info = False

            sum_auprc.append(valid_auprc)
            sum_auc.append(valid_auc)
            sum_acc.append(valid_acc)
            sum_f1.append(valid_f1)
            sum_tp.append(valid_tp)

            y_true, y_pred, y_score, y_index = predict(
                modules["model"],
                modules["test_loader_list"],
                configs,
                ckpt,
            )
            test_acc, cf_matrix, test_auprc, test_f1, test_auc = calculate_metrics(y_true, y_pred, y_score)
            test_tp = cf_matrix[1, 1]

            with open(str(configs["logfile"]), "a") as f:
                print(
                    "Test AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}".format(
                        test_auprc, test_auc, test_acc, test_f1, test_tp
                    ),
                    file=f,
                    flush=True,
                )

            split_name = "test_fold" if cross_validation else "test_single_fold"
            test_metric_rows.append(
                make_test_metric_row(
                    configs=configs,
                    repeat=repeat_idx,
                    fold=fold,
                    split_name=split_name,
                    y_true=y_true,
                    acc=test_acc,
                    cf_matrix=cf_matrix,
                    auprc=test_auprc,
                    f1=test_f1,
                    auc=test_auc,
                    checkpoint=ckpt,
                )
            )
            save_test_metrics_csv(configs, test_metric_rows)

            if cross_validation:
                train_result = pred_to_df(int(fold), train_result, y_index, y_true, y_score)

        if cross_validation and train_result is not None:
            score_col = [f"score_{i}" for i in range(num_folds)]
            train_result["avg_score"] = train_result[score_col].mean(axis=1)
            train_result["pred_label"] = train_result.apply(
                lambda row: 1 if row["avg_score"] > 0.5 else 0,
                axis=1,
            )
            score_avg_performance(train_result, score_col, str(configs["logfile"]))

            ensemble_y_true = train_result["Label"].to_numpy()
            ensemble_y_score = train_result["avg_score"].to_numpy()
            ensemble_y_pred = train_result["pred_label"].to_numpy()
            ensemble_acc, ensemble_cf_matrix, ensemble_auprc, ensemble_f1, ensemble_auc = calculate_metrics(
                ensemble_y_true,
                ensemble_y_pred,
                ensemble_y_score,
            )
            test_metric_rows.append(
                make_test_metric_row(
                    configs=configs,
                    repeat=repeat_idx,
                    fold="all",
                    split_name=f"{num_folds}_folds_avg_score",
                    y_true=ensemble_y_true,
                    acc=ensemble_acc,
                    cf_matrix=ensemble_cf_matrix,
                    auprc=ensemble_auprc,
                    f1=ensemble_f1,
                    auc=ensemble_auc,
                    checkpoint="average_of_fold_test_scores",
                )
            )
            csv_path = save_test_metrics_csv(configs, test_metric_rows)
            with open(str(configs["logfile"]), "a") as f:
                print(f"Test metrics CSV saved to: {csv_path}", file=f, flush=True)

    with open(str(configs["logfile"]), "a") as f:
        print(
            "Avg AUPRC:{:.4f}±{:.4f}, AUROC:{:.4f}±{:.4f}, ACC:{:.4f}±{:.4f}, "
            "F1:{:.4f}±{:.4f}, TP:{:.1f}±{:.1f}".format(
                np.nanmean(sum_auprc),
                np.nanstd(sum_auprc),
                np.nanmean(sum_auc),
                np.nanstd(sum_auc),
                np.nanmean(sum_acc),
                np.nanstd(sum_acc),
                np.nanmean(sum_f1),
                np.nanstd(sum_f1),
                np.nanmean(sum_tp),
                np.nanstd(sum_tp),
            ),
            file=f,
            flush=True,
        )


# =============================================================================
# 7. CLI: paths and runtime only
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CGMega runner with separated data processing. "
            "Only paths/runtime options are exposed by CLI. "
            "Training/model/data hyperparameters are fixed in DEFAULT_CONFIG."
        )
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Cancer matrix data directory, e.g. /path/to/data/Breast_Cancer_Matrix",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Root data directory containing CPDB/ and cancer matrix directories. Default: parent of data_dir.",
    )
    parser.add_argument(
        "--ppi_dir",
        default=None,
        help="Directory containing <ppi>_matrix.csv. Default: data_root/<ppi>.",
    )
    parser.add_argument(
        "--cgmega_data_path",
        required=True,
        help="Absolute path to cgmega_data.py. This path is normally provided by run.sh.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--logging_dir",
        required=True,
        help="Directory for log txt and test metrics csv.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name used for log_name and output subdirectory, e.g. MCF7_CPDB.",
    )

    parser.add_argument(
        "--gpu",
        default=None,
        help="Internal GPU id. Use 0 when CUDA_VISIBLE_DEVICES is already set. If omitted, use CPU.",
    )
    parser.add_argument("--cv", dest="cv", action="store_true", default=True)
    parser.add_argument("--no_cv", dest="cv", action="store_false")
    parser.add_argument("--load_data", dest="load_data", action="store_true", default=None)
    parser.add_argument("--rebuild_data", dest="load_data", action="store_false")

    return parser.parse_args()


def build_config(args) -> Dict[str, object]:
    configs = dict(DEFAULT_CONFIG)

    # Only path/runtime values from CLI override DEFAULT_CONFIG.
    # Architecture/training/data-processing hyperparameters remain code-level variables.
    configs["data_dir"] = args.data_dir
    configs["data_root"] = args.data_root
    configs["ppi_dir"] = args.ppi_dir
    configs["cgmega_data_path"] = args.cgmega_data_path
    configs["out_dir"] = args.output_dir
    configs["log_dir"] = args.logging_dir
    configs["log_name"] = args.dataset

    if args.load_data is not None:
        configs["load_data"] = args.load_data

    if args.gpu is not None:
        configs["device"] = f"cuda:{args.gpu}"
    else:
        configs["device"] = "cpu"

    log_dir = Path(str(configs["log_dir"]))
    out_dir = Path(str(configs["out_dir"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs["logfile"] = str(log_dir / f"{configs['log_name']}.txt")

    if configs["graph"] == "dual":
        raise ValueError("graph=dual requires DualGATRes and is intentionally removed from this minimal CGMega script.")
    if configs["graph"] not in {"ppi", "plusc"}:
        raise ValueError("This minimal CGMega script only supports graph='ppi' or graph='plusc'.")
    if configs["ppi"] in {None, "None", ""}:
        raise ValueError("Minimal CGMega requires a configured PPI value, usually CPDB.")

    return configs


def main():
    args = parse_args()
    configs = build_config(args)
    data_module = load_cgmega_data_module(str(configs["cgmega_data_path"]))
    set_seed(int(configs["random_seed"]))

    print("============================================================")
    print("CGMega started")
    print("Data processing is isolated in cgmega_data.py")
    print("Model architecture receives only ModelInputSpec(in_channels, edge_dim)")
    print(f"data_dir: {configs['data_dir']}")
    print(f"cgmega_data_path: {configs['cgmega_data_path']}")
    print(f"data_root: {configs['data_root'] or data_module.get_data_root(configs)}")
    print(f"ppi_dir: {configs['ppi_dir'] or (data_module.get_data_root(configs) / str(configs['ppi']))}")
    print(f"out_dir: {configs['out_dir']}")
    print(f"logfile: {configs['logfile']}")
    print(f"device: {configs['device']}")
    print(f"cross_validation: {args.cv}")
    print(f"load_data: {configs['load_data']}")
    print("============================================================")

    dataset = data_module.get_data(configs=configs, stable=bool(configs["stable"]))
    cv_train(dataset=dataset, configs=configs, cross_validation=bool(args.cv))


if __name__ == "__main__":
    main()

