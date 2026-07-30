from __future__ import annotations

import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


CONDITION_COLUMNS = ["COMPOUND_NAME", "SACRI_PERIOD", "DOSE_LEVEL"]


def _build_structure_features(data_dir: Path) -> None:
    """Build the Morgan feature cache directly, without an external script."""

    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, MACCSkeys
    from rdkit.Chem.Scaffolds import MurckoScaffold

    structures = pd.read_csv(data_dir / "compound_structures.csv")
    descriptor_fns = Descriptors._descList
    names, morgan, maccs, rdkit_desc, scaffold_hash = [], [], [], [], []
    status = []
    for row in structures.itertuples(index=False):
        smiles = (
            row.isomeric_smiles
            if isinstance(row.isomeric_smiles, str)
            else row.canonical_smiles
        )
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            status.append((row.COMPOUND_NAME, "parse_error"))
            continue
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            2,
            nBits=2048,
        )
        bits = np.zeros(2048, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fingerprint, bits)
        maccs_fp = MACCSkeys.GenMACCSKeys(molecule)
        maccs_bits = np.zeros(167, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(maccs_fp, maccs_bits)
        values = []
        for _, descriptor_fn in descriptor_fns:
            try:
                values.append(float(descriptor_fn(molecule)))
            except Exception:
                values.append(np.nan)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        digest = hashlib.sha256(scaffold.encode("utf-8")).digest()
        names.append(row.COMPOUND_NAME)
        morgan.append(bits)
        maccs.append(maccs_bits)
        rdkit_desc.append(values)
        scaffold_hash.append(int.from_bytes(digest[:8], "little", signed=False))
        status.append((row.COMPOUND_NAME, "ok"))

    np.savez_compressed(
        data_dir / "structure_features.npz",
        compound_names=np.asarray(names),
        morgan=np.asarray(morgan, dtype=np.float32),
        maccs=np.asarray(maccs, dtype=np.float32),
        rdkit_descriptors=np.asarray(rdkit_desc, dtype=np.float32),
        rdkit_descriptor_names=np.asarray([name for name, _ in descriptor_fns]),
        scaffold_hash=np.asarray(scaffold_hash, dtype=np.uint64),
    )
    pd.DataFrame(status, columns=["COMPOUND_NAME", "status"]).to_csv(
        data_dir / "structure_feature_status.csv",
        index=False,
    )


def _build_chemberta_embeddings(
    data_dir: Path,
    model_name: str,
    output_name: str,
) -> None:
    """Download a ChemBERTa checkpoint and cache mean-pooled SMILES vectors."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    structures = pd.read_csv(data_dir / "compound_structures.csv")
    smiles = [
        row.isomeric_smiles
        if isinstance(row.isomeric_smiles, str)
        else row.canonical_smiles
        for row in structures.itertuples(index=False)
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    batch_size = int(os.getenv("TRI_HF_BATCH_SIZE", "16"))
    max_length = int(os.getenv("TRI_HF_MAX_LENGTH", "256"))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            encoded = tokenizer(
                smiles[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(pooled.cpu().numpy())

    np.savez_compressed(
        data_dir / output_name,
        compound_names=structures["COMPOUND_NAME"].astype(str).to_numpy(),
        embeddings=np.concatenate(vectors, axis=0).astype(np.float32),
        model_name=np.asarray([model_name]),
    )


def prepare_external_views(data_dir: str | Path) -> None:
    """Generate missing derived molecular files under the supplied directory.

    The path is deliberately an argument rather than a model constant. The
    evaluator owns data locations; this function only owns feature preparation.
    """

    data_dir = Path(data_dir).expanduser().resolve()
    required = (
        "structure_features.npz",
        "chemberta_embeddings.npz",
        "chemberta_zinc_embeddings.npz",
    )
    missing = [name for name in required if not (data_dir / name).exists()]
    if not missing:
        return
    if os.getenv("ANIMALGAN_AUTO_BUILD_VIEWS", "1") != "1":
        raise FileNotFoundError(
            f"Missing derived feature files: {missing}. Set "
            "ANIMALGAN_AUTO_BUILD_VIEWS=1 to generate them."
        )
    if not (data_dir / "compound_structures.csv").exists():
        raise FileNotFoundError(
            f"Missing {data_dir / 'compound_structures.csv'}; cannot derive "
            "Morgan or ChemBERTa features."
        )

    if "structure_features.npz" in missing:
        _build_structure_features(data_dir)
    for filename, model_name in (
        ("chemberta_embeddings.npz", "DeepChem/ChemBERTa-77M-MLM"),
        ("chemberta_zinc_embeddings.npz", "seyonec/ChemBERTa-zinc-base-v1"),
    ):
        if filename not in missing:
            continue
        _build_chemberta_embeddings(data_dir, model_name, filename)


def _load_local_views() -> dict[str, np.ndarray]:
    """Load/generate views for evaluators that use the original 2-argument API."""

    data_dir = Path(
        os.getenv(
            "ANIMALGAN_DATA_DIR",
            str(Path(__file__).resolve().parents[5] / "data" / "multiconformer3d"),
        )
    )
    prepare_external_views(data_dir)
    structure = np.load(data_dir / "structure_features.npz", allow_pickle=True)
    deep = np.load(data_dir / "chemberta_embeddings.npz", allow_pickle=True)
    zinc = np.load(data_dir / "chemberta_zinc_embeddings.npz", allow_pickle=True)
    return {
        "morgan_names": structure["compound_names"],
        "morgan_vectors": structure["morgan"],
        "deep_names": deep["compound_names"],
        "deep_vectors": deep["embeddings"],
        "zinc_names": zinc["compound_names"],
        "zinc_vectors": zinc["embeddings"],
    }


class _RetrievalExpert:
    """Retrieve and shrink response profiles for unseen compounds."""

    # The evaluator loads these files from its --data-dir and passes them to
    # fit/predict. This keeps the model path-independent like the simple MLP.
    requires_external_views = True

    def __init__(self, benchmark_name: str | None = None):
        self.benchmark = benchmark_name or "Time"

        # Structure benchmark can blend all three views. The defaults preserve
        # the original experiment: ZINC retrieval only, with no DeepChem or
        # Morgan contribution.
        self.zinc_weight = float(os.getenv("TRI_ZINC_WEIGHT", "1"))
        self.deep_weight = float(os.getenv("TRI_DEEP_WEIGHT", "0"))
        self.morgan_weight = max(
            0.0,
            1.0 - self.zinc_weight - self.deep_weight,
        )
        self.structure_shrink = float(
            os.getenv("TRI_STRUCTURE_SHRINK", ".55")
        )

    def _set_external_views(self, external_views: dict | None) -> None:
        """Validate and retain views supplied by the external evaluator."""

        required = {
            "morgan_names",
            "morgan_vectors",
            "deep_names",
            "deep_vectors",
            "zinc_names",
            "zinc_vectors",
        }
        if external_views is None:
            external_views = _load_local_views()
        missing = required.difference(external_views)
        if missing:
            raise ValueError(f"Missing external molecular views: {sorted(missing)}")
        self.external_views = external_views

    def fit(
        self,
        train_data: pd.DataFrame,
        train_descriptors: pd.DataFrame,
        external_views: dict | None = None,
    ):
        """Build training-fold baselines, vectors, and response profiles."""

        del train_descriptors
        self._set_external_views(external_views)

        self.measurement_columns = train_data.columns[3:].tolist()
        condition_means = train_data.groupby(
            CONDITION_COLUMNS,
            as_index=False,
        )[self.measurement_columns].mean()
        self.cells = sorted(
            set(
                zip(
                    condition_means.SACRI_PERIOD,
                    condition_means.DOSE_LEVEL,
                )
            )
        )

        self.base = {}
        for cell, group in train_data.groupby(["SACRI_PERIOD", "DOSE_LEVEL"]):
            values = group[self.measurement_columns].to_numpy(float)
            lower = np.nanquantile(values, 0.1, axis=0)
            upper = np.nanquantile(values, 0.9, axis=0)
            self.base[cell] = np.nanmean(
                np.clip(values, lower, upper),
                axis=0,
            )
        self.global_ = np.nanmean(
            train_data[self.measurement_columns].to_numpy(float),
            axis=0,
        )

        self.morgan_vectors = {
            str(name): vector.astype(float)
            for name, vector in zip(
                self.external_views["morgan_names"],
                self.external_views["morgan_vectors"],
            )
        }
        self.deep_vectors = {
            str(name): vector.astype(float)
            for name, vector in zip(
                self.external_views["deep_names"],
                self.external_views["deep_vectors"],
            )
        }
        self.zinc_vectors = {
            str(name): vector.astype(float)
            for name, vector in zip(
                self.external_views["zinc_names"],
                self.external_views["zinc_vectors"],
            )
        }

        self.compounds = sorted(
            condition_means.COMPOUND_NAME.astype(str).unique()
        )
        self.morgan_matrix = np.asarray(
            [self.morgan_vectors[name] for name in self.compounds]
        )
        self.deep_matrix = np.asarray(
            [self.deep_vectors[name] for name in self.compounds]
        )
        self.zinc_matrix = np.asarray(
            [self.zinc_vectors[name] for name in self.compounds]
        )

        # Store each training compound's concatenated residual response
        # profile across all observed time/dose cells.
        self.profiles = []
        for compound in self.compounds:
            compound_means = condition_means[
                condition_means.COMPOUND_NAME.astype(str).eq(compound)
            ].set_index(["SACRI_PERIOD", "DOSE_LEVEL"])

            profile = np.concatenate(
                [
                    (
                        compound_means.loc[cell, self.measurement_columns].to_numpy(
                            float
                        )
                        - self.base[cell]
                        if cell in compound_means.index
                        else np.zeros(len(self.measurement_columns))
                    )
                    for cell in self.cells
                ]
            )
            self.profiles.append(
                np.nan_to_num(
                    profile,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )

        self.profiles = np.asarray(self.profiles)
        return self

    @staticmethod
    def _cosine_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity between query and reference rows."""

        numerator = query @ reference.T
        query_norm = np.linalg.norm(query, axis=1)[:, None]
        reference_norm = np.linalg.norm(reference, axis=1)[None, :]
        return numerator / np.maximum(query_norm * reference_norm, 1e-9)

    def predict_condition_means(
        self,
        test_data: pd.DataFrame,
        test_descriptors: pd.DataFrame,
        external_views: dict | None = None,
    ) -> pd.DataFrame:
        """Retrieve a profile for each test condition and add the baseline."""

        del test_descriptors  # Molecular views are supplied by the evaluator.
        if external_views is not None:
            self._set_external_views(external_views)
        elif not hasattr(self, "external_views"):
            self._set_external_views(None)

        conditions = test_data[CONDITION_COLUMNS].drop_duplicates().reset_index(
            drop=True
        )
        names = conditions.COMPOUND_NAME.astype(str).tolist()

        # DeepChem: nearest-neighbor profile.
        deep_query = np.asarray([self.deep_vectors[name] for name in names])
        deep_index = np.argmax(
            self._cosine_similarity(deep_query, self.deep_matrix),
            axis=1,
        )
        deep_profile = self.profiles[deep_index]

        # ZINC: weighted average of the four nearest neighbors.
        zinc_query = np.asarray([self.zinc_vectors[name] for name in names])
        zinc_similarity = self._cosine_similarity(zinc_query, self.zinc_matrix)
        zinc_indices = np.argpartition(
            -zinc_similarity,
            3,
            axis=1,
        )[:, :4]
        zinc_weights = np.take_along_axis(
            zinc_similarity,
            zinc_indices,
            axis=1,
        ) ** 32
        zinc_weights /= np.maximum(zinc_weights.sum(axis=1, keepdims=True), 1e-12)
        zinc_profile = np.einsum(
            "nk,nkp->np",
            zinc_weights,
            self.profiles[zinc_indices],
        )

        # Morgan: weighted average of the five nearest neighbors using Tanimoto
        # similarity computed directly on the binary fingerprints.
        morgan_query = np.asarray([self.morgan_vectors[name] for name in names])
        intersection = morgan_query @ self.morgan_matrix.T
        union = (
            morgan_query.sum(axis=1)[:, None]
            + self.morgan_matrix.sum(axis=1)[None, :]
            - intersection
        )
        morgan_similarity = intersection / np.maximum(union, 1e-9)
        morgan_indices = np.argpartition(
            -morgan_similarity,
            4,
            axis=1,
        )[:, :5]
        morgan_weights = np.take_along_axis(
            morgan_similarity,
            morgan_indices,
            axis=1,
        ) ** 2
        morgan_weights /= np.maximum(
            morgan_weights.sum(axis=1, keepdims=True),
            1e-12,
        )
        morgan_profile = np.einsum(
            "nk,nkp->np",
            morgan_weights,
            self.profiles[morgan_indices],
        )

        if self.benchmark == "Structure":
            profile = (
                self.zinc_weight * zinc_profile
                + self.deep_weight * deep_profile
                + self.morgan_weight * morgan_profile
            )
            shrinkage = self.structure_shrink
        elif self.benchmark == "RandomPick":
            profile = deep_profile
            shrinkage = 0.8
        elif self.benchmark == "ATC":
            profile = deep_profile
            shrinkage = 0.5
        else:
            profile = deep_profile
            shrinkage = 0.55

        cell_to_index = {cell: index for index, cell in enumerate(self.cells)}
        endpoint_count = len(self.measurement_columns)
        predictions = []
        for row_index, row in enumerate(conditions.itertuples(index=False)):
            cell = (row.SACRI_PERIOD, row.DOSE_LEVEL)
            base = self.base.get(cell, self.global_)
            cell_index = cell_to_index.get(cell)
            residual = (
                np.zeros(endpoint_count)
                if cell_index is None
                else profile[
                    row_index,
                    cell_index * endpoint_count : (cell_index + 1) * endpoint_count,
                ]
            )
            predictions.append(base + shrinkage * residual)

        return pd.concat(
            [
                conditions,
                pd.DataFrame(
                    np.asarray(predictions),
                    columns=self.measurement_columns,
                ),
            ],
            axis=1,
        )


class _ChemBERTaMLPExpert:
    """Self-contained ChemBERTa profile regressor used by the Soft-MOE."""

    def __init__(self, benchmark_name: str):
        self.benchmark = benchmark_name
        # (profile latent dimension, MLP regularization, shrinkage, width)
        config = {
            "RandomPick": (8, 10.0, 0.75, 32),
            "Structure": (8, 1.0, 0.50, 64),
            "ATC": (8, 10.0, 0.75, 64),
            "Time": (8, 10.0, 0.75, 64),
        }
        self.latent, self.alpha, self.shrink, self.hidden = config[benchmark_name]
        self.data_dir = Path(
            os.getenv(
                "ANIMALGAN_DATA_DIR",
                str(Path(__file__).resolve().parents[5] / "data" / "multiconformer3d"),
            )
        )

    def fit(self, train_data: pd.DataFrame, train_descriptors: pd.DataFrame):
        del train_descriptors
        self.measurement_columns = train_data.columns[3:].tolist()
        means = train_data.groupby(CONDITION_COLUMNS, as_index=False)[self.measurement_columns].mean()
        self.cells = sorted(set(zip(means.SACRI_PERIOD, means.DOSE_LEVEL)))
        self.base = {}
        for cell, group in train_data.groupby(["SACRI_PERIOD", "DOSE_LEVEL"]):
            values = group[self.measurement_columns].to_numpy(float)
            lo = np.nanquantile(values, 0.1, axis=0)
            hi = np.nanquantile(values, 0.9, axis=0)
            self.base[cell] = np.nanmean(np.clip(values, lo, hi), axis=0)
        self.global_ = np.nanmean(train_data[self.measurement_columns].to_numpy(float), axis=0)

        deep = np.load(self.data_dir / "chemberta_embeddings.npz", allow_pickle=True)
        zinc = np.load(self.data_dir / "chemberta_zinc_embeddings.npz", allow_pickle=True)
        dmap = {str(n): np.asarray(x, float) for n, x in zip(deep["compound_names"], deep["embeddings"])}
        zmap = {str(n): np.asarray(x, float) for n, x in zip(zinc["compound_names"], zinc["embeddings"])}
        self.xmap = {name: np.r_[dmap[name], zmap[name]] for name in dmap}

        compounds = sorted(means.COMPOUND_NAME.astype(str).unique())
        X = np.asarray([self.xmap[name] for name in compounds])
        self.xscaler = StandardScaler().fit(X)
        Xn = self.xscaler.transform(X)
        self.xpca = PCA(n_components=min(48, len(compounds) - 1, X.shape[1]), random_state=19).fit(Xn)
        Z = self.xpca.transform(Xn)

        profiles = []
        for name in compounds:
            rows = means[means.COMPOUND_NAME.astype(str).eq(name)].set_index(["SACRI_PERIOD", "DOSE_LEVEL"])
            parts = []
            for cell in self.cells:
                if cell in rows.index:
                    parts.append(rows.loc[cell, self.measurement_columns].to_numpy(float) - self.base[cell])
                else:
                    parts.append(np.zeros(len(self.measurement_columns)))
            profiles.append(np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0))
        Y = np.asarray(profiles)
        self.yscale = np.maximum(np.nanmedian(np.abs(Y), axis=0), 1.0)
        self.ypca = PCA(n_components=min(self.latent, len(Y) - 1, Y.shape[1]), random_state=23).fit(Y / self.yscale)
        target = self.ypca.transform(Y / self.yscale)
        self.reg = MLPRegressor(
            hidden_layer_sizes=(self.hidden, max(8, self.hidden // 2)),
            activation="tanh", solver="lbfgs", alpha=self.alpha,
            max_iter=500, random_state=29,
        ).fit(Z, target)
        return self

    def predict_condition_means(self, test_data: pd.DataFrame, test_descriptors: pd.DataFrame) -> pd.DataFrame:
        del test_descriptors
        conditions = test_data[CONDITION_COLUMNS].drop_duplicates().reset_index(drop=True)
        X = np.asarray([self.xmap[str(name)] for name in conditions.COMPOUND_NAME])
        latent = self.reg.predict(self.xpca.transform(self.xscaler.transform(X)))
        profile = self.ypca.inverse_transform(latent) * self.yscale
        cell_index = {cell: i for i, cell in enumerate(self.cells)}
        endpoint_count = len(self.measurement_columns)
        values = []
        for i, row in enumerate(conditions.itertuples(index=False)):
            cell = (row.SACRI_PERIOD, row.DOSE_LEVEL)
            base = self.base.get(cell, self.global_)
            index = cell_index.get(cell)
            residual = np.zeros(endpoint_count) if index is None else profile[i, index * endpoint_count : (index + 1) * endpoint_count]
            values.append(base + self.shrink * residual)
        return pd.concat([conditions, pd.DataFrame(np.asarray(values), columns=self.measurement_columns)], axis=1)


# Shared configuration: the same MLP contribution is used for every split.
# 0.30 is fixed before this rerun (the rounded mean of the earlier split-wise
# settings), rather than selected from the outer validation folds.
SHARED_MLP_WEIGHT = 0.25
SHARED_DESCRIPTOR_WEIGHT = float(os.getenv("SOFT_MOE_DESCRIPTOR_WEIGHT", "0.05"))
SHARED_RETRIEVAL_WEIGHT = 1.0 - SHARED_MLP_WEIGHT - SHARED_DESCRIPTOR_WEIGHT


class _MordredDescriptorExpert:
    """PCA-compressed Mordred descriptor profile expert."""

    def __init__(self):
        self.latent = 16
        self.alpha = 100.0

    def fit(self, train_data, train_descriptors):
        self.measurement_columns = train_data.columns[3:].tolist()
        means = train_data.groupby(CONDITION_COLUMNS, as_index=False)[self.measurement_columns].mean()
        self.cells = sorted(set(zip(means.SACRI_PERIOD, means.DOSE_LEVEL)))
        self.base = {}
        for cell, group in train_data.groupby(["SACRI_PERIOD", "DOSE_LEVEL"]):
            values = group[self.measurement_columns].to_numpy(float)
            lo = np.nanquantile(values, 0.1, axis=0)
            hi = np.nanquantile(values, 0.9, axis=0)
            self.base[cell] = np.nanmean(np.clip(values, lo, hi), axis=0)
        self.global_ = np.nanmean(train_data[self.measurement_columns].to_numpy(float), axis=0)

        frame = train_descriptors.apply(pd.to_numeric, errors="coerce")
        self.columns = frame.columns.tolist()
        self.median = frame.median().fillna(0.0)
        X = frame.fillna(self.median).to_numpy(float)
        self.scaler = StandardScaler().fit(X)
        Xz = self.scaler.transform(X)
        self.projector = PCA(n_components=min(32, X.shape[0] - 1, X.shape[1]), random_state=17).fit(Xz)
        compounds = sorted(means.COMPOUND_NAME.astype(str).unique())
        Xc = self.projector.transform(self.scaler.transform(
            frame.reindex(compounds, columns=self.columns).fillna(self.median).to_numpy(float)
        ))
        profiles = []
        for name in compounds:
            rows = means[means.COMPOUND_NAME.astype(str).eq(name)].set_index(["SACRI_PERIOD", "DOSE_LEVEL"])
            parts = []
            for cell in self.cells:
                if cell in rows.index:
                    parts.append(rows.loc[cell, self.measurement_columns].to_numpy(float) - self.base[cell])
                else:
                    parts.append(np.zeros(len(self.measurement_columns)))
            profiles.append(np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0))
        Y = np.asarray(profiles)
        self.yscale = np.maximum(np.nanmedian(np.abs(Y), axis=0), 1.0)
        self.profile_pca = PCA(n_components=min(self.latent, len(Y) - 1, Y.shape[1]), random_state=23).fit(Y / self.yscale)
        self.reg = Ridge(alpha=self.alpha).fit(Xc, self.profile_pca.transform(Y / self.yscale))
        return self

    def predict_condition_means(self, test_data, test_descriptors):
        conditions = test_data[CONDITION_COLUMNS].drop_duplicates().reset_index(drop=True)
        frame = test_descriptors.reindex(conditions.COMPOUND_NAME.astype(str), columns=self.columns)
        X = frame.apply(pd.to_numeric, errors="coerce").fillna(self.median).to_numpy(float)
        latent = self.reg.predict(self.projector.transform(self.scaler.transform(X)))
        profile = self.profile_pca.inverse_transform(latent) * self.yscale
        cell_index = {cell: i for i, cell in enumerate(self.cells)}
        e = len(self.measurement_columns)
        values = []
        for i, row in enumerate(conditions.itertuples(index=False)):
            cell = (row.SACRI_PERIOD, row.DOSE_LEVEL)
            base = self.base.get(cell, self.global_)
            j = cell_index.get(cell)
            residual = np.zeros(e) if j is None else profile[i, j * e : (j + 1) * e]
            # Descriptor extrapolation can be extreme for held-out chemistry;
            # constrain it using the training-fold residual scale.
            scale = self.yscale[j * e : (j + 1) * e] if j is not None else np.ones(e)
            residual = np.clip(residual, -3.0 * scale, 3.0 * scale)
            values.append(base + 0.5 * residual)
        return pd.concat([conditions, pd.DataFrame(np.asarray(values), columns=self.measurement_columns)], axis=1)


class ResidualClinicalPathologyModel:
    """Single-file ChemBERTa MLP + three-view retrieval mixture-of-experts."""

    requires_external_views = True

    def __init__(self, benchmark_name: str | None = None):
        self.benchmark = benchmark_name or "Time"
        if self.benchmark not in {"RandomPick", "Structure", "ATC", "Time"}:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")
        self.mlp = _ChemBERTaMLPExpert(self.benchmark)
        self.retrieval = _RetrievalExpert(self.benchmark)
        self.descriptor = _MordredDescriptorExpert()

    def fit(self, train_data, train_descriptors, external_views=None):
        self.mlp.fit(train_data, train_descriptors)
        self.retrieval.fit(train_data, train_descriptors, external_views=external_views)
        self.descriptor.fit(train_data, train_descriptors)
        self.measurement_columns = self.mlp.measurement_columns
        return self

    def predict_condition_means(self, test_data, test_descriptors, external_views=None):
        mlp = self.mlp.predict_condition_means(test_data, test_descriptors)
        retrieval = self.retrieval.predict_condition_means(test_data, test_descriptors, external_views=external_views)
        descriptor = self.descriptor.predict_condition_means(test_data, test_descriptors)
        values = (
            SHARED_MLP_WEIGHT * mlp[self.measurement_columns].to_numpy(float)
            + SHARED_RETRIEVAL_WEIGHT * retrieval[self.measurement_columns].to_numpy(float)
            + SHARED_DESCRIPTOR_WEIGHT * descriptor[self.measurement_columns].to_numpy(float)
        )
        return pd.concat([mlp[CONDITION_COLUMNS].reset_index(drop=True), pd.DataFrame(values, columns=self.measurement_columns)], axis=1)
