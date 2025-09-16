from __future__ import annotations

import gc
import importlib
import os
import sys
import warnings
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import scipy
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import FLOAT_DTYPES

if TYPE_CHECKING:
    import pandas as pd


# TODO: merge into TabPFnv2 codebase
class FixedSafePowerTransformer(PowerTransformer):
    """Fixed version of safe power THAT FOLLOWS BASIC SKLEARN STANDARD ANS THUS DOES NOT HAVE A BUG WHEN CLONING
    WHY IS THIS SO HARD?
    """

    def __init__(
        self,
        variance_threshold: float = 1e-3,
        large_value_threshold: float = 100,
        method="yeo-johnson",
        standardize=True,
        copy=True,
    ):
        super().__init__(method=method, standardize=standardize, copy=copy)
        self.variance_threshold = variance_threshold
        self.large_value_threshold = large_value_threshold

        self.revert_indices_ = None

    def _find_features_to_revert_because_of_failure(
        self,
        transformed_X: np.ndarray,
    ) -> None:
        # Calculate the variance for each feature in the transformed data
        variances = np.nanvar(transformed_X, axis=0)

        # Identify features where the variance is not close to 1
        mask = np.abs(variances - 1) > self.variance_threshold
        non_unit_variance_indices = np.where(mask)[0]

        # Identify features with values greater than the large_value_threshold
        large_value_indices = np.any(transformed_X > self.large_value_threshold, axis=0)
        large_value_indices = np.nonzero(large_value_indices)[0]

        # Identify features to revert based on either condition
        self.revert_indices_ = np.unique(
            np.concatenate([non_unit_variance_indices, large_value_indices]),
        )

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"overflow encountered",
                    category=RuntimeWarning,
                )
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except scipy.optimize._optimize.BracketError:
            return np.nan

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        if np.isnan(lmbda):
            return x

        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore

    def _revert_failed_features(
        self,
        transformed_X: np.ndarray,
        original_X: np.ndarray,
    ) -> np.ndarray:
        # Replace these features with the original features
        if self.revert_indices_ and (self.revert_indices_) > 0:
            transformed_X[:, self.revert_indices_] = original_X[:, self.revert_indices_]

        return transformed_X

    def fit(self, X: np.ndarray, y: Any | None = None) -> FixedSafePowerTransformer:
        super().fit(X, y)

        # Check and revert features as necessary
        self._find_features_to_revert_because_of_failure(super().transform(X))  # type: ignore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed_X = super().transform(X)
        return self._revert_failed_features(transformed_X, X)  # type: ignore


# TODO: merge into codebase or remove KDITransformer from search space
def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
    """Check inputs before fit and transform."""
    return self._validate_data(
        X,
        reset=in_fit,
        accept_sparse=False,
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )

def _revert_chunked_attention() -> None:  # noqa: D401 – imperative form
    """Undo **all** monkey‑patches applied to ``MultiHeadAttention``.

    Idempotent – safe to call even if no patch is active.
    """
    mha_mod = importlib.import_module("tabpfn.model.multi_head_attention")
    cls = getattr(mha_mod, "MultiHeadAttention")

    if getattr(cls, "_is_chunk_patched", False):
        if hasattr(cls, "_original_forward"):
            cls.forward = cls._original_forward  # type: ignore[attr-defined]
            delattr(cls, "_original_forward")
        if hasattr(cls, "_original_compute_attention_heads"):
            cls.compute_attention_heads = cls._original_compute_attention_heads  # type: ignore[attr-defined]
            delattr(cls, "_original_compute_attention_heads")
        cls._is_chunk_patched = False  # type: ignore[attr-defined]


def _apply_chunk_patch(*, chunk_size: int = 2048, batch_size: int = 20_000) -> None:
    """(Re‑)apply the requested chunk‑attention monkey‑patch."""
    _revert_chunked_attention()

    from tabpfn.extensions.replace_chunk_attn_computation import enable_chunked_attention as _patch
    _patch(chunk_size=chunk_size, batch_size=batch_size)


def _clear_cuda() -> None:
    """Aggressively free *all* reserved CUDA memory."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# -----------------------------------------------------------------------------
# Main AutoGluon model class
# -----------------------------------------------------------------------------

class TabPFNV2Model(AbstractModel):
    ag_key = "CUSTOMTABPFNV2"
    ag_name = "CustomTabPFNv2"

    # sensible defaults exposed as class constants
    _FAST_CHUNK_SIZE = 4096
    _MEM_CHUNK_SIZE = 256
    _MEM_BATCH_SIZE = 10_000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._cat_features = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        self._cat_indices = []

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        # This converts categorical features to numeric via stateful label encoding.
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

            # Detect/set cat features and indices
            if self._cat_features is None:
                self._cat_features = self._feature_generator.features_in[:]
            self._cat_indices = [X.columns.get_loc(col) for col in self._cat_features]

        return X

    # FIXME: What is the minimal model artifact?
    #  If zeroshot, maybe we don't save weights for each fold in bag and instead load from a single weights file?
    # FIXME: Crashes during model download if bagging with parallel fit.
    #  Consider adopting same download logic as TabPFNMix which doesn't crash during model download.
    # FIXME: Maybe support child_oof somehow with using only one model and being smart about inference time?
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        *,
        fast_chunk_size: int | None = None,
        mem_chunk_size: int | None = None,
        mem_batch_size: int | None = None,
        **kwargs,
    ):
        import sys
        sys.path.insert(0, os.environ.get("PYTHONPATH") + "/src")
        
        from tabpfn.model import preprocessing

        preprocessing.SafePowerTransformer = FixedSafePowerTransformer
        preprocessing.KDITransformerWithNaN._check_inputs = _check_inputs

        from tabpfn.classifier import TabPFNClassifier
        from tabpfn.regressor import TabPFNRegressor
        from tabpfn.model.loading import resolve_model_path

        def _build_model() -> Any:
            ag_params = self._get_ag_params()
            max_classes = ag_params.get("max_classes")
            is_classification = self.problem_type in ["binary", "multiclass"]

            if is_classification:
                if max_classes is not None and self.num_classes > max_classes:
                    raise AssertionError(
                        f"Max allowed classes is {max_classes}, but found {self.num_classes}.",
                    )
                model_base: type = TabPFNClassifier
            else:
                model_base = TabPFNRegressor

            device = "cuda" if num_gpus else "cpu"

            hps = {**self._get_model_params()}
            hps.update(
                {
                    "device": device,
                    "n_jobs": num_cpus,
                    "random_state": 42,
                    "categorical_features_indices": self._cat_indices,
                    "ignore_pretraining_limits": True,
                }
            )

            # checkpoint resolution
            _, model_dir, _, _ = resolve_model_path(
                model_path=None,
                which="classifier" if is_classification else "regressor",
            )
            if is_classification:
                if path := hps.pop("classification_model_path", None):
                    hps["model_path"] = model_dir / path
                hps.pop("regression_model_path", None)
            else:
                if path := hps.pop("regression_model_path", None):
                    hps["model_path"] = model_dir / path
                hps.pop("classification_model_path", None)

            # inference_config cleanup (power→safepower)
            inference_config = {
                _k: v for k, v in hps.items() if k.startswith("inference_config/") and (_k := k.split("/")[-1])
            }
            if inference_config:
                hps["inference_config"] = inference_config
            for k in list(hps):
                if k.startswith("inference_config/"):
                    del hps[k]

            # TODO: remove power from search space and TabPFNv2 codebase
            # Power transform can fail. To avoid this, make all power be safepower instead.
            if "PREPROCESS_TRANSFORMS" in inference_config:
                safe_config = []
                for preprocessing_dict in inference_config["PREPROCESS_TRANSFORMS"]:
                    if preprocessing_dict["name"] == "power":
                        preprocessing_dict["name"] = "safepower"
                    safe_config.append(preprocessing_dict)
                inference_config["PREPROCESS_TRANSFORMS"] = safe_config
            if "REGRESSION_Y_PREPROCESS_TRANSFORMS" in inference_config:
                safe_config = []
                for preprocessing_name in inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"]:
                    if preprocessing_name == "power":
                        preprocessing_name = "safepower"
                    safe_config.append(preprocessing_name)
                inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = safe_config


            # RF‑PFN ensemble handling
            n_ensemble_repeats = hps.pop("n_ensemble_repeats", None)
            if hps.pop("model_type", "no") == "dt_pfn":
                from tabrepo.benchmark.models.ag.tabpfnv2.rfpfn import (
                    RandomForestTabPFNClassifier,
                    RandomForestTabPFNRegressor,
                )

                hps["n_estimators"] = 1
                rf_base = RandomForestTabPFNClassifier if is_classification else RandomForestTabPFNRegressor
                model = rf_base(
                    tabpfn=model_base(**hps),
                    categorical_features=self._cat_indices,
                    n_estimators=n_ensemble_repeats,
                )
            else:
                if n_ensemble_repeats is not None:
                    hps["n_estimators"] = n_ensemble_repeats
                model = model_base(**hps)

            return model

        # allow per‑call overrides
        fast_chunk_size = fast_chunk_size or self._FAST_CHUNK_SIZE
        mem_chunk_size = mem_chunk_size or self._MEM_CHUNK_SIZE
        mem_batch_size = mem_batch_size or self._MEM_BATCH_SIZE

        # single preprocessing pass (can reuse across retries)
        X_proc = self.preprocess(X, is_train=True)

        _clear_cuda()
        _apply_chunk_patch(chunk_size=mem_chunk_size, batch_size=mem_batch_size)
        self.model = _build_model().fit(X=X_proc, y=y)

    def _get_default_resources(self) -> tuple[int, int]:
        from autogluon.common.utils.resource_utils import ResourceManager
        from torch.cuda import is_available

        num_cpus = ResourceManager.get_cpu_count_psutil()
        num_gpus = 1 if is_available() else 0
        return num_cpus, num_gpus


    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)


    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]


    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            },
        )
        return default_auxiliary_params


    def _ag_params(self) -> set:
        return {"max_classes"}


    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
