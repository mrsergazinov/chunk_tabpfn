from __future__ import annotations

import json
from pathlib import Path
import requests
from typing import Literal
from typing_extensions import Self

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix
import pandas as pd
import yaml

from tabrepo.loaders import Paths
from tabrepo.repository.evaluation_repository import EvaluationRepository
from tabrepo.nips2025_utils.generate_repo import generate_repo_from_results_lst
from tabrepo.benchmark.result import BaselineResult
from tabrepo.nips2025_utils.method_processor import get_info_from_result, load_raw


class MethodMetadata:
    def __init__(
        self,
        method: str,
        *,
        artifact_name: str = None,
        date: str | None = None,
        method_type: Literal["config", "baseline", "portfolio"] = "config",
        name_suffix: str | None = None,
        ag_key: str | None = None,
        config_default: str | None = None,
        can_hpo: bool | None = None,
        compute: Literal["cpu", "gpu"] = "cpu",
        is_bag: bool = False,
        has_raw: bool = False,
        has_processed: bool = False,
        has_results: bool = False,
        upload_as_public: bool = False,
    ):
        self.method = method
        if artifact_name is None:
            artifact_name = method
        self.artifact_name = artifact_name
        self.date = date
        self.method_type = method_type
        self.ag_key = ag_key
        self.name_suffix = name_suffix
        self.config_default = config_default
        self.compute = compute
        self.is_bag = is_bag
        self.has_raw = has_raw
        self.has_processed = has_processed
        self.has_results = has_results
        self.upload_as_public = upload_as_public
        if can_hpo is None:
            can_hpo = self.method_type == "config"
        self.can_hpo = can_hpo

        assert isinstance(self.method, str) and len(self.method) > 0
        assert isinstance(self.artifact_name, str) and len(self.artifact_name) > 0
        assert self.method_type in ["config", "baseline", "portfolio"]
        assert self.compute in ["cpu", "gpu"]

    # TODO: Also support baseline methods
    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult],
        method: str | None = None,
        artifact_name: str | None = None,
        config_default: str | None = None,
        compute: Literal["cpu", "gpu"] | None = None,
    ) -> Self:
        result_lst_dict = []

        for r in results_lst:
            cur_result = get_info_from_result(result=r)
            result_lst_dict.append(cur_result)
        result_df = pd.DataFrame(result_lst_dict)

        unique_method_types = result_df["method_type"].unique()
        assert len(unique_method_types) == 1
        method_type = unique_method_types[0]

        assert method_type == "config"

        unique_model_types = result_df["model_type"].unique()
        assert len(unique_model_types) == 1

        unique_num_gpus = result_df["num_gpus"].unique()
        assert len(unique_num_gpus) == 1
        num_gpus = unique_num_gpus[0]

        if compute is None:
            compute: Literal["cpu", "gpu"] = "cpu" if num_gpus == 0 else "gpu"

        unique_ag_key = result_df["ag_key"].unique()
        assert len(unique_ag_key) == 1
        ag_key = unique_ag_key[0]

        is_bag = bool(result_df["is_bag"].any())

        unique_name_prefix = result_df["name_prefix"].unique()
        assert len(unique_name_prefix) == 1
        name_prefix = unique_name_prefix[0]

        unique_methods = result_df["framework"].unique()
        if len(unique_methods) == 1:
            _config_default = unique_methods[0]
            can_hpo = False
        else:
            _config_default = None
            can_hpo = True
        if config_default is None:
            config_default = _config_default

        if method is None:
            method = name_prefix

        if artifact_name is None:
            artifact_name = method

        _method_metadata = cls(
            method=method,
            artifact_name=artifact_name,
            method_type=method_type,
            compute=compute,
            config_default=config_default,
            ag_key=ag_key,
            can_hpo=can_hpo,
            is_bag=is_bag,
            has_raw=True,
            has_processed=True,
            has_results=True,
        )

        return _method_metadata

    @property
    def has_configs_hyperparameters(self) -> bool:
        return self.method_type == "config"

    @property
    def _path_root(self) -> Path:
        return Paths.artifacts_root_cache_tabarena

    @property
    def path_cache_root(self) -> Path:
        return Paths._tabarena_root_cache

    @property
    def path(self) -> Path:
        return self._path_root / self.artifact_name / "methods" / self.method

    @property
    def path_raw(self) -> Path:
        return self.path / "raw"

    @property
    def path_processed(self) -> Path:
        return self.path / "processed"

    @property
    def path_processed_holdout(self) -> Path:
        return self.path / "processed_holdout"

    @property
    def path_results(self) -> Path:
        return self.path / "results"

    @property
    def path_results_holdout(self) -> Path:
        return self.path_results / "holdout"

    def path_results_hpo(self, holdout: bool = False) -> Path:
        path_prefix = self.path_results_holdout if holdout else self.path_results
        return path_prefix / "hpo_results.parquet"

    def path_results_model(self, holdout: bool = False) -> Path:
        path_prefix = self.path_results_holdout if holdout else self.path_results
        return path_prefix / "model_results.parquet"

    def path_results_portfolio(self, holdout: bool = False) -> Path:
        path_prefix = self.path_results_holdout if holdout else self.path_results
        return path_prefix / "portfolio_results.parquet"

    def relative_to_cache_root(self, path: Path) -> Path:
        return path.relative_to(self.path_cache_root)

    def relative_to_root(self, path: Path) -> Path:
        return path.relative_to(self._path_root)

    def relative_to_method(self, path: Path) -> Path:
        return path.relative_to(self.path)

    def to_s3_cache_loc(self, path: Path, s3_cache_root: str) -> str:
        path_suffix = self.relative_to_cache_root(path=path)
        s3_cache_path = f"{s3_cache_root}/{path_suffix}"
        return s3_cache_path

    def load_model_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_model(holdout=holdout))

    def load_hpo_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_hpo(holdout=holdout))

    def load_portfolio_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_portfolio(holdout=holdout))

    def path_configs_hyperparameters(self, holdout: bool = False) -> Path:
        if holdout:
            path_processed = self.path_processed_holdout
        else:
            path_processed = self.path_processed
        path_configs_hyperparameters = path_processed / "configs_hyperparameters.json"
        return path_configs_hyperparameters

    def load_configs_hyperparameters(self, holdout: bool = False) -> dict[str, dict]:
        with open(self.path_configs_hyperparameters(holdout=holdout), "r") as f:
            out = json.load(f)
        return out

    def download_configs_hyperparameters(self, s3_cache_root: str, holdout: bool = False):
        path_local = self.path_configs_hyperparameters(holdout=holdout)
        s3_path_loc = self.to_s3_cache_loc(path=path_local, s3_cache_root=s3_cache_root)
        self._download_file(url=s3_path_loc, local_path=path_local)

    def upload_configs_hyperparameters(self, s3_cache_root: str, holdout: bool = False):
        path_local = self.path_configs_hyperparameters(holdout=holdout)
        s3_path_loc = self.to_s3_cache_loc(path=path_local, s3_cache_root=s3_cache_root)
        self._upload_file(file_name=path_local, s3_path=s3_path_loc)

    def _upload_file(self, file_name: str | Path, s3_path: str):
        import boto3

        kwargs = {}
        if self.upload_as_public:
            kwargs = {"ExtraArgs": {"ACL": "public-read"}}
        bucket, prefix = s3_path_to_bucket_prefix(s3_path)

        # Upload the file
        s3_client = boto3.client("s3")
        s3_client.upload_file(Filename=file_name, Bucket=bucket, Key=prefix, **kwargs)

    def _download_file(self, url: str, local_path: str | Path):
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP request errors

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

    def load_raw(
        self,
        path_raw: str | Path = None,
        engine: str = "ray",
        as_holdout: bool = False,
    ) -> list[BaselineResult]:
        """
        Loads the raw results artifacts from all `results.pkl` files in the `path_raw` directory

        Parameters
        ----------
        path_raw
        engine
        as_holdout

        Returns
        -------

        """
        if path_raw is None:
            path_raw = self.path_raw
        return load_raw(path_raw=path_raw, engine=engine, as_holdout=as_holdout)

    def generate_repo(
        self,
        results_lst: list = None,
        task_metadata: pd.DataFrame = None,
        cache: bool = False,
        engine: str = "ray",
    ) -> EvaluationRepository:
        if results_lst is None:
            results_lst = self.load_raw(engine=engine)
        path_processed = self.path_processed
        name_suffix = self.name_suffix

        repo: EvaluationRepository = generate_repo_from_results_lst(
            results_lst=results_lst,
            task_metadata=task_metadata,
            name_suffix=name_suffix,
        )

        if cache:
            repo.to_dir(path_processed)
        return repo

    @property
    def path_metadata(self) -> Path:
        return self.path / "metadata.yaml"

    def to_yaml(self, path: Path | str = None):
        if path is None:
            path = self.path_metadata
        assert str(path).endswith(".yaml")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as outfile:
            yaml.dump(self.__dict__, outfile, default_flow_style=False)

    @classmethod
    def from_yaml(
        cls,
        path: Path | str = None,
        method: str = None,
        artifact_name: str = None,
    ) -> Self:
        if path is None:
            assert method is not None, f"method must be specified if path is not specified"
            assert artifact_name is not None, f"artifact_name must be specified if path is not specified"
            path = Paths._tabarena_root_cache / "artifacts" / artifact_name / "methods" / method / "metadata.yaml"

        assert str(path).endswith(".yaml")
        with open(path, 'r') as file:
            kwargs = yaml.safe_load(file)
        return cls(**kwargs)

    def cache_raw(
        self,
        results_lst: list[BaselineResult],
    ):
        path = self.path_raw
        n_results = len(results_lst)
        for i, result in enumerate(results_lst):
            if i % 100 == 0:
                print(f"{i + 1}/{n_results}")
            result.to_dir(path=path)
