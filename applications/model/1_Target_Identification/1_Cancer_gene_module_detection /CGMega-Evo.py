#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CGMega training with a configuration-centered characteristic ranker.

Raw data access remains isolated in the dynamically loaded cgmega_data.py.
This module consumes only its processed PyG dataset.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import random
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch.nn import Dropout, MaxPool1d
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import dropout_adj

try:
    from transformers import get_linear_schedule_with_warmup
except Exception:
    get_linear_schedule_with_warmup = None


def load_cgmega_data_module(cgmega_data_path: str) -> ModuleType:
    """Dynamically load the externally supplied separated data module."""
    module_path = Path(cgmega_data_path).expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"cgmega_data.py does not exist: {module_path}")
    if module_path.name != "cgmega_data.py":
        raise ValueError(f"Expected a file named cgmega_data.py, got: {module_path}")
    spec = importlib.util.spec_from_file_location("cgmega_data", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cgmega_data"] = module
    spec.loader.exec_module(module)
    for name in ("CancerDataset", "get_data", "get_data_root"):
        if not hasattr(module, name):
            raise AttributeError(f"{module_path} is missing required symbol: {name}")
    return module


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
    # Every labeled negative is retained; ranking adds hard-negative pressure.
    "sample_neg": 1.0,
    "sample_pos": 1.0,
    "sample_rate": 1.0,
}

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
    "data_dir": None,
    "data_root": None,
    "ppi_dir": None,
    "cgmega_data_path": None,
    "joint": False,
    "load_data": True,
    "log_dir": None,
    "log_name": None,
    "logfile": None,
    "model": "config_centered_characteristic_ranker",
    "neighbors": [-1, -1],
    "out_dir": None,
    "reverse": False,
    "device": "cpu",
    "fold": 0,
    **CORE_HYPERPARAMETERS,
    **DATA_GRAPH_HYPERPARAMETERS,
}

HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2
CHAR_EMBED_DIM = 64
CHAR_FREQUENCIES = 12
FINGERPRINT_DIM = 32
INTERACTION_DIM = 24


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
    with open(filename, "a") as stream:
        for key, value in configs.items():
            print(key, ":", value, file=stream, flush=True)


@dataclass(frozen=True)
class ModelInputSpec:
    """Only dataset-derived shape metadata visible to the architecture."""

    in_channels: int
    edge_dim: int


def infer_model_input_spec(dataset: Any) -> ModelInputSpec:
    data = dataset[0]
    edge_dim = getattr(data, "edge_dim", None)
    if edge_dim is None:
        edge_attr = getattr(data, "edge_attr", None)
        edge_dim = 1 if edge_attr is None or edge_attr.dim() == 1 else edge_attr.size(-1)
    return ModelInputSpec(int(dataset.num_node_features), int(edge_dim))


class CGMega(t.nn.Module):
    """Protected CGMega anchor plus a non-recursive characteristic correction."""

    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, residual,
                 devices_available):
        super(CGMega, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate
        self.residual = residual
        mid_channels = in_channels + hidden_channels if residual else hidden_channels

        self.convs = t.nn.ModuleList([
            TransformerConv(
                in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate,
                edge_dim=edge_dim, concat=False, beta=True,
            ),
            TransformerConv(
                mid_channels, hidden_channels, heads=heads, dropout=attn_drop_rate,
                edge_dim=edge_dim, concat=True, beta=True,
            ),
        ])
        self.ln1 = LayerNorm(in_channels=mid_channels)
        self.ln2 = LayerNorm(in_channels=hidden_channels * heads)
        self.pool = MaxPool1d(2, 2)
        self.dropout = Dropout(drop_rate)
        self.lins = t.nn.ModuleList([
            Linear(int(hidden_channels * heads / 2), HIDDEN_DIM, weight_initializer="kaiming_uniform"),
            Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform"),
        ])

        # All feature columns, including condensed Hi-C coordinates, enter jointly.
        self.feature_skip = t.nn.Linear(in_channels, CHAR_EMBED_DIM)
        self.feature_in = t.nn.Linear(in_channels, CHAR_EMBED_DIM)
        self.feature_out = t.nn.Linear(CHAR_EMBED_DIM, CHAR_EMBED_DIM)
        self.frequency_raw = t.nn.Parameter(torch.randn(CHAR_FREQUENCIES, CHAR_EMBED_DIM) * 0.15)
        self.fingerprint_map = t.nn.Linear(2 * CHAR_FREQUENCIES, FINGERPRINT_DIM, bias=False)
        self.z_interaction = t.nn.Linear(CHAR_EMBED_DIM, INTERACTION_DIM, bias=False)
        self.q_interaction = t.nn.Linear(FINGERPRINT_DIM, INTERACTION_DIM, bias=False)
        self.correction_head = t.nn.Linear(
            2 * FINGERPRINT_DIM + INTERACTION_DIM, 1, bias=False
        )

        self.gamma_raw = t.nn.Parameter(torch.tensor(-3.0))
        self.temperature_raw = t.nn.Parameter(torch.tensor(-0.6931471806))
        self.offset_raw = t.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("lambda_recall", torch.tensor(0.0))
        self.register_buffer("lambda_fpr", torch.tensor(0.0))
        self.register_buffer("frequency_radius", torch.linspace(0.5, 1.5, CHAR_FREQUENCIES))
        self.last_state: Dict[str, torch.Tensor] = {}
        self.debug_shapes = os.environ.get("DEBUG_SHAPES", "0") == "1"
        self.to(self.devices_available)

    @staticmethod
    def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _edge_weight(data, dtype: torch.dtype) -> torch.Tensor:
        edge_count = data.edge_index.size(1)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            return torch.ones(edge_count, dtype=dtype, device=data.edge_index.device)
        weight = edge_attr if edge_attr.dim() == 1 else edge_attr[:, 0]
        weight = torch.nan_to_num(
            weight.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        return weight.clamp_min(0.0)

    def _anchor_logits(self, data) -> torch.Tensor:
        edge_index, edge_attr = dropout_adj(
            data.edge_index, getattr(data, "edge_attr", None), p=self.drop_rate,
            force_undirected=True, training=self.training,
        )
        residual = data.x
        x = self.convs[0](data.x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE, inplace=True)
        x = torch.cat((x, residual), dim=1) if self.residual else x
        x = self.ln1(x)
        edge_index, edge_attr = dropout_adj(
            data.edge_index, getattr(data, "edge_attr", None), p=self.drop_rate,
            force_undirected=True, training=self.training,
        )
        x = self.convs[1](x, edge_index, edge_attr)
        x = F.leaky_relu(self.ln2(x), negative_slope=LEAKY_SLOPE)
        x = self.pool(x.unsqueeze(1)).squeeze(1)
        x = self.dropout(self.lins[0](x).relu())
        return self.lins[1](x).squeeze(-1)

    def _characteristic_fingerprint(self, data, z: torch.Tensor) -> torch.Tensor:
        """Sparse observed-minus-configuration-null characteristic moments."""
        num_nodes = z.size(0)
        src, dst = data.edge_index
        weight = self._edge_weight(data, z.dtype)
        omega = 2.0 * torch.tanh(self.frequency_raw)
        phase = z @ omega.transpose(0, 1)
        phase_cos, phase_sin = torch.cos(phase), torch.sin(phase)

        edge_delta = phase[src] - phase[dst]
        edge_pair = torch.stack((torch.cos(edge_delta), torch.sin(edge_delta)), dim=-1)
        observed_sum = z.new_zeros((num_nodes, CHAR_FREQUENCIES, 2))
        observed_sum.index_add_(0, dst, edge_pair * weight[:, None, None])
        strength = z.new_zeros(num_nodes)
        strength.index_add_(0, dst, weight)
        observed = observed_sum / strength.clamp_min(1e-8)[:, None, None]

        endpoint_degree = getattr(data, "ccr_degree", strength).to(dtype=z.dtype)
        endpoint_mass = endpoint_degree.sum().clamp_min(1e-8)
        global_cos = (endpoint_degree[:, None] * phase_cos).sum(0) / endpoint_mass
        global_sin = (endpoint_degree[:, None] * phase_sin).sum(0) / endpoint_mass
        null_real = phase_cos * global_cos + phase_sin * global_sin
        null_imag = phase_cos * global_sin - phase_sin * global_cos
        null = torch.stack((null_real, null_imag), dim=-1)

        weight_square = z.new_zeros(num_nodes)
        weight_square.index_add_(0, dst, weight.square())
        effective_n = strength.square() / weight_square.clamp_min(1e-8)
        nonisolated = (strength > 0).to(z.dtype)
        shrinkage = effective_n / (effective_n + 2.0)
        fingerprint = ((observed - null) * shrinkage[:, None, None]).reshape(
            num_nodes, 2 * CHAR_FREQUENCIES
        )

        # Ridge-project out constant, linear, and quadratic log-degree effects.
        degree_z = getattr(data, "ccr_degree_z", torch.log1p(endpoint_degree)).to(z.dtype)
        nuisance = torch.stack(
            (torch.ones_like(degree_z), degree_z, degree_z.square()), dim=1
        )
        weighted_nuisance = nuisance * nonisolated[:, None]
        gram = nuisance.transpose(0, 1) @ weighted_nuisance
        gram = gram + 1e-3 * torch.eye(3, dtype=z.dtype, device=z.device)
        rhs = weighted_nuisance.transpose(0, 1) @ fingerprint
        beta = _solve_small_linear_system(gram, rhs)
        return (fingerprint - nuisance @ beta) * nonisolated[:, None]

    def frequency_regularization(self) -> torch.Tensor:
        omega = 2.0 * torch.tanh(self.frequency_raw)
        norms = torch.linalg.vector_norm(omega, dim=1)
        radial = (norms - self.frequency_radius).square().mean()
        unit = omega / norms.clamp_min(1e-6)[:, None]
        similarity = unit @ unit.transpose(0, 1)
        similarity = similarity - torch.diag_embed(torch.diagonal(similarity))
        diversity = similarity.square().sum() / max(
            1, CHAR_FREQUENCIES * (CHAR_FREQUENCIES - 1)
        )
        return radial + diversity

    def forward(self, data):
        batch = data[0].to(self.devices_available)
        anchor_logits = self._anchor_logits(batch)
        z = self.feature_skip(batch.x) + self.feature_out(F.gelu(self.feature_in(batch.x)))
        z = self._rms_norm(z)
        fingerprint = self._characteristic_fingerprint(batch, z)
        q = torch.tanh(self.fingerprint_map(fingerprint))
        interaction = self.z_interaction(z) * self.q_interaction(q)
        correction = self.correction_head(
            torch.cat((q, q.abs(), interaction), dim=1)
        ).squeeze(-1)

        gamma = 2.0 * torch.sigmoid(self.gamma_raw)
        ordering = anchor_logits.detach() + gamma * correction
        temperature = 0.5 + 1.5 * torch.sigmoid(self.temperature_raw)
        offset = 3.0 * torch.tanh(self.offset_raw)
        deployed_logits = (ordering - offset) / temperature
        self.last_state = {
            "anchor_logits": anchor_logits,
            "correction": correction,
            "raw_ordering": ordering,
            "deployed_logits": deployed_logits,
            "temperature": temperature,
            "offset": offset,
            "gamma": gamma,
        }
        if self.debug_shapes:
            expected = (batch.x.size(0),)
            if deployed_logits.shape != expected:
                raise RuntimeError(f"Expected logits {expected}, got {tuple(deployed_logits.shape)}")
            if not torch.isfinite(deployed_logits).all():
                raise FloatingPointError("Non-finite characteristic-ranker logits")
        return torch.sigmoid(deployed_logits).unsqueeze(-1)


class ConfigCenteredRankerLoss(t.nn.Module):
    """Proper BCE, robust pair ranking, and anchor-relative operating guards."""

    def __init__(self, model: CGMega):
        super().__init__()
        self._model_ref = weakref.ref(model)
        self.step_count = 0
        self.warmup_steps = 1
        self.last_violations = None

    @property
    def model(self) -> CGMega:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Ranker model was released before its loss")
        return model

    def set_warmup_steps(self, steps: int) -> None:
        self.warmup_steps = max(1, int(steps))

    @staticmethod
    def _ranking_loss(ordering: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        positive_mask = target > 0.5
        negative_mask = ~positive_mask
        if not positive_mask.any() or not negative_mask.any():
            return ordering.sum() * 0.0
        center = ordering.detach().median()
        scale = (ordering.detach() - center).abs().median().clamp_min(1e-3)
        standardized = (ordering - center) / scale
        positive = standardized[positive_mask]
        negative = standardized[negative_mask]
        violation = 0.25 * F.softplus(
            (0.5 + negative[None, :] - positive[:, None]) / 0.25
        )
        adversarial = torch.softmax(
            (violation.detach().clamp_max(4.0) / 0.75).reshape(-1), dim=0
        ).reshape_as(violation)
        uniform = torch.full_like(violation, 1.0 / violation.numel())
        return ((0.5 * uniform + 0.5 * adversarial) * violation).sum()

    def forward(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del score
        model = self.model
        if not model.last_state:
            raise RuntimeError("Model forward must run before ranker loss")
        count = target.numel()
        deployed = model.last_state["deployed_logits"][:count]
        if not model.training:
            return F.binary_cross_entropy_with_logits(deployed, target)

        anchor = model.last_state["anchor_logits"][:count]
        ordering = model.last_state["raw_ordering"][:count]
        correction = model.last_state["correction"][:count]
        anchor_loss = F.binary_cross_entropy_with_logits(anchor, target)
        classification_loss = F.binary_cross_entropy_with_logits(deployed, target)
        ranking_loss = self._ranking_loss(ordering, target)

        positive, negative = target, 1.0 - target
        smooth_final = torch.sigmoid(deployed / 0.25)
        smooth_anchor = torch.sigmoid(anchor.detach() / 0.25)
        positive_count = positive.sum().clamp_min(1.0)
        negative_count = negative.sum().clamp_min(1.0)
        recall = (positive * smooth_final).sum() / positive_count
        anchor_recall = ((positive * smooth_anchor).sum() / positive_count).detach()
        fpr = (negative * smooth_final).sum() / negative_count
        anchor_fpr = ((negative * smooth_anchor).sum() / negative_count).detach()
        recall_violation = F.relu(anchor_recall - 0.05 - recall)
        fpr_violation = F.relu(fpr - anchor_fpr - 0.03)
        self.last_violations = torch.stack(
            (recall_violation.detach(), fpr_violation.detach())
        )
        ramp = min(1.0, float(self.step_count) / float(self.warmup_steps))
        operating = (
            model.lambda_recall * recall_violation
            + model.lambda_fpr * fpr_violation
            + 2.0 * (recall_violation.square() + fpr_violation.square())
        )
        return (
            anchor_loss
            + classification_loss
            + ramp * (0.2 * ranking_loss + operating)
            + 0.01 * correction.square().mean()
            + 0.002 * model.frequency_regularization()
        )

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        if self.last_violations is not None:
            self.model.lambda_recall.add_(0.1 * self.last_violations[0]).clamp_(0.0, 5.0)
            self.model.lambda_fpr.add_(0.1 * self.last_violations[1]).clamp_(0.0, 5.0)
        self.step_count += 1


def safe_div(numerator, denominator) -> float:
    return float(numerator) / float(denominator) if float(denominator) != 0 else np.nan


def calculate_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if not np.all(np.isin(np.unique(y_true), [0, 1, 0.0, 1.0])):
        raise ValueError(f"y_true contains non-binary values: {y_true[:20]}")
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)
    if not np.all(np.isin(np.unique(y_pred), [0, 1, 0.0, 1.0])):
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
    tn, fp, fn, tp = cf_matrix.ravel()
    recall = safe_div(tp, tp + fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "Precision": safe_div(tp, tp + fp),
        "MCC": safe_div(tp * tn - fp * fn, denominator),
        "Recall": recall,
        "Sensitivity": recall,
        "Specificity": safe_div(tn, tn + fp),
        "NPV": safe_div(tn, tn + fn),
        "FPR": safe_div(fp, fp + tn),
        "FNR": safe_div(fn, fn + tp),
    }


def get_test_metrics_csv_path(configs: Dict[str, object]) -> str:
    base, _ = os.path.splitext(str(configs["logfile"]))
    return base + "_test_metrics.csv"


def save_test_metrics_csv(configs: Dict[str, object], test_metric_rows: List[Dict[str, object]]) -> str:
    out_csv = get_test_metrics_csv_path(configs)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(test_metric_rows).to_csv(out_csv, index=False)
    return out_csv


def make_test_metric_row(configs, repeat, fold, split_name, y_true, acc, cf_matrix,
                         auprc, f1, auc, checkpoint=None):
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = cf_matrix.ravel()
    extended = calculate_extended_binary_metrics(cf_matrix)
    return {
        "split": split_name,
        "repeat": repeat,
        "fold": fold,
        "model": configs.get("model", "config_centered_characteristic_ranker"),
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


def drop_samples(dataset: Any, fold: int, sample_neg: float = 0.0,
                 sample_pos: float = 1.0, num_samples: int = 0,
                 random_seed: int = 42) -> List[int]:
    if sample_neg == 1 and sample_pos == 1:
        return []
    train_idx = dataset.get_idx_split(fold)["train"]
    negative, positive = [], []
    for item in train_idx:
        index = int(item.item())
        if dataset[0].y[index][0]:
            negative.append(index)
        if dataset[0].y[index][1]:
            positive.append(index)
    random.seed(random_seed)
    if num_samples:
        neg_keep = int(num_samples * len(negative) / max(1, len(negative) + len(positive)))
        pos_keep = int(num_samples) - neg_keep
        drop_negative = random.sample(negative, max(0, len(negative) - neg_keep))
        drop_positive = random.sample(positive, max(0, len(positive) - pos_keep))
    else:
        drop_negative = random.sample(negative, int(len(negative) * (1 - sample_neg)))
        drop_positive = random.sample(positive, int(len(positive) * (1 - sample_pos)))
    dropped = drop_negative + drop_positive
    print(f"Negatives: {len(negative) - len(drop_negative)}, Positives: {len(positive) - len(drop_positive)}")
    dataset[0].train_mask[dropped, fold] = False
    return dropped


def build_model(configs: Dict[str, object], input_spec: ModelInputSpec) -> CGMega:
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
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _solve_small_linear_system(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve the tiny degree-residualization system without relying on CUDA MAGMA."""
    if gram.is_cuda or rhs.is_cuda:
        solution = torch.linalg.solve(gram.cpu(), rhs.cpu())
        return solution.to(device=rhs.device, dtype=rhs.dtype)
    return torch.linalg.solve(gram, rhs)


def _attach_label_free_graph_statistics(data) -> None:
    """Attach dataset-derived degree metadata copied by NeighborLoader."""
    if hasattr(data, "ccr_degree") and hasattr(data, "ccr_degree_z"):
        return
    src, dst = data.edge_index
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        weight = torch.ones(dst.numel(), dtype=data.x.dtype, device=dst.device)
    else:
        weight = edge_attr if edge_attr.dim() == 1 else edge_attr[:, 0]
        weight = torch.nan_to_num(weight.to(data.x.dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    degree = data.x.new_zeros(data.x.size(0))
    degree.index_add_(0, dst, weight)
    log_degree = torch.log1p(degree)
    active = degree > 0
    if active.any():
        mean = log_degree[active].mean()
        std = log_degree[active].std(unbiased=False).clamp_min(1e-6)
        degree_z = (log_degree - mean) / std
    else:
        degree_z = torch.zeros_like(degree)
    data.ccr_degree = degree
    data.ccr_degree_z = degree_z


def get_training_modules(configs: Dict[str, object], dataset: Any):
    fold = int(configs["fold"])
    print("Drop", 1 - float(configs["sample_neg"]), "of negative train samples and",
          1 - float(configs["sample_pos"]), "of positive train samples")
    drop_idx = drop_samples(
        dataset, fold, sample_neg=float(configs["sample_neg"]),
        sample_pos=float(configs["sample_pos"]), random_seed=int(configs["random_seed"]),
    )
    if float(configs["sample_rate"]) < 1:
        drop_idx += drop_samples(
            dataset, fold, sample_neg=float(configs["sample_rate"]),
            sample_pos=float(configs["sample_rate"]), random_seed=int(configs["random_seed"]),
        )
    elif float(configs["sample_rate"]) > 1:
        drop_idx = drop_samples(
            dataset, fold, num_samples=int(configs["sample_rate"]),
            random_seed=int(configs["random_seed"]),
        )

    data = dataset[0]
    _attach_label_free_graph_statistics(data)
    neighbors, batch_size = configs["neighbors"], int(configs["batch_size"])
    train_loader = NeighborLoader(
        data, num_neighbors=neighbors, batch_size=batch_size, directed=False,
        input_nodes=data.train_mask[:, fold], shuffle=True,
    )
    valid_loader = NeighborLoader(
        data, num_neighbors=neighbors, batch_size=batch_size, directed=False,
        input_nodes=data.valid_mask[:, fold], shuffle=False,
    )
    test_loader = NeighborLoader(
        data, num_neighbors=neighbors, batch_size=batch_size, directed=False,
        input_nodes=data.test_mask, shuffle=False,
    )
    model = build_model(configs, infer_model_input_spec(dataset))
    optimizer = t.optim.AdamW(
        model.parameters(), lr=float(configs["lr"]),
        weight_decay=float(configs["weight_decay"]),
    )
    num_train_nodes = int(data.train_mask[:, fold].sum().item())
    steps_per_epoch = int(np.ceil(num_train_nodes / batch_size))
    total_steps = steps_per_epoch * int(configs["num_epochs"])
    scheduler = build_scheduler(optimizer, total_steps)
    loss_func = ConfigCenteredRankerLoss(model)
    loss_func.set_warmup_steps(max(1, total_steps // 10))
    return {
        "dataset": dataset,
        "model": model,
        "loss_func": loss_func,
        "train_loader_list": [train_loader],
        "valid_loader_list": [valid_loader],
        "test_loader_list": [test_loader],
        "optimizer": optimizer,
        "scheduler": scheduler,
        "drop_idx": drop_idx,
    }


def run_loader(model, loader_list, device: str, loss_func=None, return_loss: bool = False):
    model.eval()
    y_true, y_pred, y_score, y_index = [], [], [], []
    total_loss, steps = 0.0, 0
    for data_tuple in zip(*loader_list):
        size = data_tuple[0].batch_size
        with torch.no_grad():
            out = model(data_tuple)[:size].view(-1)
            true_lab = data_tuple[0].y[:size][:, 1].to(device).float()
            pred_lab = (out >= 0.5).to(true_lab.dtype)
            if return_loss and loss_func is not None:
                total_loss += loss_func(out, true_lab).item()
                steps += 1
        y_true.extend(true_lab.detach().cpu().numpy().tolist())
        y_pred.extend(pred_lab.detach().cpu().numpy().tolist())
        y_score.extend(out.detach().cpu().numpy().tolist())
        y_index.extend(data_tuple[0].pos[:size].detach().cpu().numpy().tolist())
    result = (np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score), np.asarray(y_index))
    if return_loss:
        return (*result, total_loss / max(1, steps))
    return result


def train_one_epoch(model, train_loader_list, optimizer, device: str,
                    scheduler=None, loss_func=None) -> float:
    model.train()
    total_loss, steps = 0.0, 0
    for data_tuple in zip(*train_loader_list):
        optimizer.zero_grad(set_to_none=True)
        size = data_tuple[0].batch_size
        out = model(data_tuple)[:size].view(-1)
        true_lab = data_tuple[0].y[:size][:, 1].to(device).float()
        loss = loss_func(out, true_lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if hasattr(loss_func, "after_optimizer_step"):
            loss_func.after_optimizer_step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.detach().item()
        steps += 1
    return total_loss / max(1, steps)


def train_model(modules, configs, log_name: str, fold: int, head_info: bool = False):
    logfile, device = str(configs["logfile"]), str(configs["device"])
    dataset, model = modules["dataset"], modules["model"]
    optimizer, scheduler, loss_func = modules["optimizer"], modules["scheduler"], modules["loss_func"]
    data = dataset[0]
    if head_info:
        print_config(logfile, configs)
        with open(logfile, "a") as stream:
            print("Model: config_centered_characteristic_ranker\nTrain/Valid/Test:",
                  int(data.train_mask[:, fold].sum()), int(data.valid_mask[:, fold].sum()),
                  int(data.test_mask.sum()), file=stream, flush=True)
            print(model, file=stream, flush=True)
    print("Start Training")
    best_auprc, best_auc, best_acc, best_f1 = -np.inf, np.nan, np.nan, np.nan
    best_tp, best_epoch, best_checkpoint, trigger_times = 0, -1, None, 0
    patience = max(1, int(configs["num_epochs"]) // 5)
    start_check_epoch = max(0, int(configs["num_epochs"]) // 10)

    for epoch in range(int(configs["num_epochs"])):
        train_loss = train_one_epoch(
            model, modules["train_loader_list"], optimizer, device,
            scheduler=scheduler, loss_func=loss_func,
        )
        y_true_train, y_pred_train, y_score_train, _ = run_loader(
            model, modules["train_loader_list"], device
        )
        train_acc, _, train_auprc, _, train_auc = calculate_metrics(
            y_true_train, y_pred_train, y_score_train
        )
        y_true, y_pred, y_score, _, valid_loss = run_loader(
            model, modules["valid_loader_list"], device,
            loss_func=loss_func, return_loss=True,
        )
        acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, "
                  f"Acc: {acc:.4f}, AUPRC: {auprc:.4f}, TP: {cf_matrix[1, 1]}, "
                  f"F1: {f1:.4f}, AUROC: {auc:.4f}, Train AUPRC: {train_auprc:.4f}, "
                  f"Train AUROC: {train_auc:.4f}, Train ACC: {train_acc:.4f}")
        if epoch >= start_check_epoch:
            if auprc < best_auprc:
                trigger_times += 1
                if trigger_times == patience:
                    print("Early Stopping")
                    break
            else:
                trigger_times = 0
                best_auprc, best_auc, best_acc, best_f1 = auprc, auc, acc, f1
                best_tp, best_epoch = int(cf_matrix[1, 1]), epoch
                best_checkpoint = {
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler else None,
                }
    if best_checkpoint is None:
        best_checkpoint = {
            "state_dict": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler else None,
        }
    model_dir = Path(str(configs["out_dir"])) / log_name
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / f"{fold}_{best_auprc:.4f}_{best_auc:.4f}_{best_tp}.pkl"
    t.save(best_checkpoint, ckpt_path)
    with open(logfile, "a") as stream:
        print("epoch {}: AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}".format(
            best_epoch, best_auprc, best_auc, best_acc, best_f1, best_tp
        ), file=stream, flush=True)
    dataset[0].train_mask[modules["drop_idx"], fold] = True
    return best_auprc, best_auc, best_acc, best_f1, best_tp, str(ckpt_path)


def predict(model, loader_list, configs: Dict[str, object], ckpt: str):
    device = str(configs["device"])
    print(f"Loading model from {ckpt} ......")
    checkpoint = t.load(ckpt, map_location=model.devices_available)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    y_true, y_pred, y_score, y_index = run_loader(model, loader_list, device)
    return (
        np.asarray(y_true).reshape(-1).astype(int),
        np.asarray(y_pred).reshape(-1).astype(int),
        np.asarray(y_score).reshape(-1).astype(float),
        np.asarray(y_index).reshape(-1),
    )


def pred_to_df(i: int, result, y_index, y_true, y_score):
    if i == 0:
        values = np.array([y_index, y_true, y_score]).T
        return pd.DataFrame(values, columns=["gene_index", "Label", f"score_{i}"])
    values = pd.DataFrame(np.array([y_index, y_score]).T,
                          columns=["gene_index", f"score_{i}"])
    return result.merge(values)


def score_avg_performance(train_result: pd.DataFrame, score_col: List[str], logfile: str) -> None:
    y_true = train_result["Label"].to_numpy()
    y_score = train_result["avg_score"].to_numpy()
    y_pred = train_result["pred_label"].to_numpy()
    acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)
    with open(logfile, "a") as stream:
        print(f"{len(score_col)}-folds AUPRC:{auprc:.4f}, AUROC:{auc:.4f}, "
              f"ACC:{acc:.4f}, F1:{f1:.4f}, TP:{cf_matrix[1, 1]:.1f}",
              file=stream, flush=True)


def cv_train(dataset: Any, configs: Dict[str, object], cross_validation: bool = True):
    log_name, num_folds = str(configs["log_name"]), int(configs["cv_folds"])
    sum_auprc, sum_auc, sum_acc, sum_f1, sum_tp = [], [], [], [], []
    test_metric_rows: List[Dict[str, object]] = []
    for repeat_idx in range(int(configs["repeat"])):
        head_info, train_result = repeat_idx == 0, None
        fold_range = range(num_folds) if cross_validation else [int(configs.get("fold", 0))]
        for fold in fold_range:
            configs["fold"] = int(fold)
            modules = get_training_modules(configs, dataset)
            valid_auprc, valid_auc, valid_acc, valid_f1, valid_tp, ckpt = train_model(
                modules, configs, log_name, int(fold), head_info=head_info
            )
            head_info = False
            sum_auprc.append(valid_auprc)
            sum_auc.append(valid_auc)
            sum_acc.append(valid_acc)
            sum_f1.append(valid_f1)
            sum_tp.append(valid_tp)
            y_true, y_pred, y_score, y_index = predict(
                modules["model"], modules["test_loader_list"], configs, ckpt
            )
            test_acc, matrix, test_auprc, test_f1, test_auc = calculate_metrics(
                y_true, y_pred, y_score
            )
            with open(str(configs["logfile"]), "a") as stream:
                print("Test AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}".format(
                    test_auprc, test_auc, test_acc, test_f1, matrix[1, 1]
                ), file=stream, flush=True)
            test_metric_rows.append(make_test_metric_row(
                configs, repeat_idx, fold,
                "test_fold" if cross_validation else "test_single_fold",
                y_true, test_acc, matrix, test_auprc, test_f1, test_auc, ckpt,
            ))
            save_test_metrics_csv(configs, test_metric_rows)
            if cross_validation:
                train_result = pred_to_df(int(fold), train_result, y_index, y_true, y_score)

        if cross_validation and train_result is not None:
            score_col = [f"score_{i}" for i in range(num_folds)]
            train_result["avg_score"] = train_result[score_col].mean(axis=1)
            train_result["pred_label"] = (train_result["avg_score"] >= 0.5).astype(int)
            score_avg_performance(train_result, score_col, str(configs["logfile"]))
            y_true = train_result["Label"].to_numpy()
            y_score = train_result["avg_score"].to_numpy()
            y_pred = train_result["pred_label"].to_numpy()
            acc, matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)
            test_metric_rows.append(make_test_metric_row(
                configs, repeat_idx, "all", f"{num_folds}_folds_avg_score",
                y_true, acc, matrix, auprc, f1, auc, "average_of_fold_test_scores",
            ))
            csv_path = save_test_metrics_csv(configs, test_metric_rows)
            with open(str(configs["logfile"]), "a") as stream:
                print(f"Test metrics CSV saved to: {csv_path}", file=stream, flush=True)
    with open(str(configs["logfile"]), "a") as stream:
        print("Avg AUPRC:{:.4f}+-{:.4f}, AUROC:{:.4f}+-{:.4f}, ACC:{:.4f}+-{:.4f}, "
              "F1:{:.4f}+-{:.4f}, TP:{:.1f}+-{:.1f}".format(
                  np.nanmean(sum_auprc), np.nanstd(sum_auprc),
                  np.nanmean(sum_auc), np.nanstd(sum_auc),
                  np.nanmean(sum_acc), np.nanstd(sum_acc),
                  np.nanmean(sum_f1), np.nanstd(sum_f1),
                  np.nanmean(sum_tp), np.nanstd(sum_tp),
              ), file=stream, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=(
        "CGMega runner with separated data processing. Only paths/runtime options "
        "are exposed by CLI. Training/model/data hyperparameters are fixed in DEFAULT_CONFIG."
    ))
    parser.add_argument("--data_dir", required=True,
                        help="Cancer matrix data directory, e.g. /path/to/data/Breast_Cancer_Matrix")
    parser.add_argument("--data_root", default=None,
                        help="Root data directory containing CPDB/ and cancer matrix directories. Default: parent of data_dir.")
    parser.add_argument("--ppi_dir", default=None,
                        help="Directory containing <ppi>_matrix.csv. Default: data_root/<ppi>.")
    parser.add_argument("--cgmega_data_path", required=True,
                        help="Absolute path to cgmega_data.py. This path is normally provided by run.sh.")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints.")
    parser.add_argument("--logging_dir", required=True,
                        help="Directory for log txt and test metrics csv.")
    parser.add_argument("--dataset", required=True,
                        help="Name used for log_name and output subdirectory, e.g. MCF7_CPDB.")
    parser.add_argument("--gpu", default=None,
                        help="Internal GPU id. Use 0 when CUDA_VISIBLE_DEVICES is already set. If omitted, use CPU.")
    parser.add_argument("--cv", dest="cv", action="store_true", default=True)
    parser.add_argument("--no_cv", dest="cv", action="store_false")
    parser.add_argument("--load_data", dest="load_data", action="store_true", default=None)
    parser.add_argument("--rebuild_data", dest="load_data", action="store_false")
    return parser.parse_args()


def build_config(args) -> Dict[str, object]:
    configs = dict(DEFAULT_CONFIG)
    configs.update({
        "data_dir": args.data_dir,
        "data_root": args.data_root,
        "ppi_dir": args.ppi_dir,
        "cgmega_data_path": args.cgmega_data_path,
        "out_dir": args.output_dir,
        "log_dir": args.logging_dir,
        "log_name": args.dataset,
        "device": f"cuda:{args.gpu}" if args.gpu is not None else "cpu",
    })
    if args.load_data is not None:
        configs["load_data"] = args.load_data
    log_dir, out_dir = Path(str(configs["log_dir"])), Path(str(configs["out_dir"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs["logfile"] = str(log_dir / f"{configs['log_name']}.txt")
    if configs["graph"] == "dual":
        raise ValueError("graph=dual requires DualGATRes and is not supported by this ranker.")
    if configs["graph"] not in {"ppi", "plusc"}:
        raise ValueError("This ranker supports graph='ppi' or graph='plusc'.")
    if configs["ppi"] in {None, "None", ""}:
        raise ValueError("A configured PPI value, usually CPDB, is required.")
    return configs


def main():
    args = parse_args()
    configs = build_config(args)
    data_module = load_cgmega_data_module(str(configs["cgmega_data_path"]))
    set_seed(int(configs["random_seed"]))
    print("============================================================")
    print("config_centered_characteristic_ranker started")
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
