from __future__ import annotations

import numpy as np


def convert_numpy_dtypes(data: dict) -> dict:
    """Converts NumPy dtypes in a dictionary to Python dtypes.
    Some hyperparameter search space's generate configs with
    numpy dtypes which aren't serializable to yaml. This fixes that.
    """
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, np.generic):
            converted_data[key] = value.item()
        elif isinstance(value, dict):
            converted_data[key] = convert_numpy_dtypes(value)
        elif isinstance(value, list):
            converted_data[key] = [
                convert_numpy_dtypes({i: v})[i] if isinstance(v, (dict, np.generic)) else v for i, v in enumerate(value)
            ]
        else:
            converted_data[key] = value
    return converted_data


def get_configs_generator_from_name(model_name: str):
    """Maps model names to their respective import functions."""
    import importlib

    name_to_import_map = {
        "CatBoost": lambda: importlib.import_module("tabrepo.models.catboost.generate").gen_catboost,
        "EBM": lambda: importlib.import_module("tabrepo.models.ebm.generate").gen_ebm,
        "ExtraTrees": lambda: importlib.import_module("tabrepo.models.extra_trees.generate").gen_extratrees,
        "FastaiMLP": lambda: importlib.import_module("tabrepo.models.fastai.generate").gen_fastai,
        "FTTransformer": lambda: importlib.import_module("tabrepo.models.ftt.generate").gen_fttransformer,
        "KNN": lambda: importlib.import_module("tabrepo.models.knn.generate").gen_knn,
        "LightGBM": lambda: importlib.import_module("tabrepo.models.lightgbm.generate").gen_lightgbm,
        "Linear": lambda: importlib.import_module("tabrepo.models.lr.generate").gen_linear,
        "ModernNCA": lambda: importlib.import_module("tabrepo.models.modernnca.generate").gen_modernnca,
        "TorchMLP": lambda: importlib.import_module("tabrepo.models.nn_torch.generate").gen_nn_torch,
        "RandomForest": lambda: importlib.import_module("tabrepo.models.random_forest.generate").gen_randomforest,
        "RealMLP": lambda: importlib.import_module("tabrepo.models.realmlp.generate").gen_realmlp,
        "TabDPT": lambda: importlib.import_module("tabrepo.models.tabdpt.generate").gen_tabdpt,
        "TabICL": lambda: importlib.import_module("tabrepo.models.tabicl.generate").gen_tabicl,
        "TabM": lambda: importlib.import_module("tabrepo.models.tabm.generate").gen_tabm,
        # "TabPFN": lambda: importlib.import_module("tabrepo.models.tabpfn.generate").gen_tabpfn, # not supported in TabArena
        "TabPFNv2": lambda: importlib.import_module("tabrepo.models.tabpfnv2.generate").gen_tabpfnv2,
        "XGBoost": lambda: importlib.import_module("tabrepo.models.xgboost.generate").gen_xgboost,
    }

    if model_name not in name_to_import_map:
        raise ValueError(f"Model name '{model_name}' is not recognized. Options are: {list(name_to_import_map.keys())}")

    return name_to_import_map[model_name]()
