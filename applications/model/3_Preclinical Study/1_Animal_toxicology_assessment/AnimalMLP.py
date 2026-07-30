from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


CONDITION_COLUMNS = ["COMPOUND_NAME", "SACRI_PERIOD", "DOSE_LEVEL"]
PERIODS = ["4 day", "8 day", "15 day", "29 day"]
DOSES = ["Low", "Middle", "High"]


@dataclass
class ResidualClinicalPathologyModel:
    """A train-only time-dose baseline with one PCA and one 64-unit MLP."""

    pca_components: int = 32
    hidden_units: int = 64
    random_state: int = 0

    targets_: list[str] | None = None
    columns_: list[str] | None = None
    median_: pd.Series | None = None
    scaler_: StandardScaler | None = None
    pca_: PCA | None = None
    model_: MLPRegressor | None = None
    baseline_: dict | None = None
    residual_scale_: np.ndarray | None = None

    def fit(self, train_data: pd.DataFrame, train_descriptors: pd.DataFrame):
        self.targets_ = train_data.columns[3:].tolist()
        means = train_data.groupby(CONDITION_COLUMNS, as_index=False)[self.targets_].mean()
        self.baseline_ = self._fit_baseline(means)
        residual = means[self.targets_].to_numpy(float) - self._baseline(means[CONDITION_COLUMNS])
        self.residual_scale_ = np.maximum(residual.std(axis=0), 1.0)

        numeric = train_descriptors.apply(pd.to_numeric, errors="coerce")
        self.columns_ = numeric.columns.tolist()
        self.median_ = numeric.median().fillna(0.0)
        x = numeric.fillna(self.median_).to_numpy(float)
        self.scaler_ = StandardScaler().fit(x)
        n_components = min(self.pca_components, x.shape[0] - 1, x.shape[1])
        self.pca_ = PCA(n_components=n_components, whiten=True, random_state=self.random_state)
        self.pca_.fit(self.scaler_.transform(x))

        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden_units,), activation="relu", solver="adam",
            alpha=0.03, learning_rate_init=3e-4, batch_size=64, max_iter=500,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=35,
            random_state=self.random_state,
        )
        self.model_.fit(self._features(means[CONDITION_COLUMNS], train_descriptors), residual / self.residual_scale_)
        return self

    def _fit_baseline(self, means: pd.DataFrame) -> dict:
        return {
            "global": means[self.targets_].mean().to_numpy(float),
            "cells": means.groupby(["SACRI_PERIOD", "DOSE_LEVEL"])[self.targets_].mean(),
        }

    def _baseline(self, conditions: pd.DataFrame) -> np.ndarray:
        out = np.tile(self.baseline_["global"], (len(conditions), 1))
        cells = self.baseline_["cells"]
        for i, row in enumerate(conditions.itertuples(index=False)):
            key = (row.SACRI_PERIOD, row.DOSE_LEVEL)
            if key in cells.index:
                out[i] = cells.loc[key, self.targets_].to_numpy(float)
        return out

    def _features(self, conditions: pd.DataFrame, descriptors: pd.DataFrame) -> np.ndarray:
        frame = descriptors.reindex(conditions["COMPOUND_NAME"].tolist(), columns=self.columns_)
        x = frame.apply(pd.to_numeric, errors="coerce").fillna(self.median_).to_numpy(float)
        z = self.pca_.transform(self.scaler_.transform(x))
        period = np.column_stack([(conditions["SACRI_PERIOD"] == item).to_numpy(float) for item in PERIODS])
        dose = np.column_stack([(conditions["DOSE_LEVEL"] == item).to_numpy(float) for item in DOSES])
        return np.hstack([z, period, dose])

    def predict_condition_means(self, test_data: pd.DataFrame, test_descriptors: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")
        conditions = test_data[CONDITION_COLUMNS].drop_duplicates().reset_index(drop=True)
        # Prevent a descriptor extrapolation from producing an implausibly
        # large profile correction; bounds use train-fold residual scale only.
        residual = self.model_.predict(self._features(conditions, test_descriptors)) * self.residual_scale_
        residual = np.clip(residual, -3.0 * self.residual_scale_, 3.0 * self.residual_scale_)
        prediction = np.maximum(self._baseline(conditions) + residual, 0.0)
        return pd.concat([conditions, pd.DataFrame(prediction, columns=self.targets_)], axis=1)
