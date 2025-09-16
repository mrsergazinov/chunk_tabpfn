from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.per_dataset_tables import get_per_dataset_tables
from tabrepo.paper.tabarena_evaluator import TabArenaEvaluator


def _get_problem_type(n_classes: int):
    if n_classes < 2:
        return "regression"
    elif n_classes == 2:
        return "binary"
    else:
        return "multiclass"


def evaluate_all(
    df_results: pd.DataFrame,
    eval_save_path: str | Path,
    df_results_holdout: pd.DataFrame = None,
    df_results_cpu: pd.DataFrame = None,
    df_results_configs: pd.DataFrame = None,
    configs_hyperparameters: dict[str, dict] = None,
    elo_bootstrap_rounds: int = 100,
    use_latex: bool = False,
):
    evaluator_kwargs = {"use_latex": use_latex}

    datasets_tabpfn = list(load_task_metadata(subset="TabPFNv2")["name"])
    datasets_tabicl = list(load_task_metadata(subset="TabICL")["name"])
    task_metadata = load_task_metadata()

    task_metadata["problem_type"] = task_metadata["NumberOfClasses"].apply(_get_problem_type)

    eval_save_path = Path(eval_save_path)

    tabicl_type = "TABICL_GPU"
    tabpfn_type = "TABPFNV2_GPU"

    portfolio_name = "TabArena ensemble (4h)"

    df_results = df_results.copy(deep=True)
    df_results["method"] = df_results["method"].map({
        "Portfolio-N200 (ensemble) (4h)": portfolio_name
    }).fillna(df_results["method"])

    if df_results_configs is not None:
        config_types_valid = df_results["config_type"].unique()
        df_results_configs_only_valid = df_results_configs[df_results_configs["config_type"].isin(config_types_valid)]
        plotter_runtime = TabArenaEvaluator(
            output_dir=eval_save_path / "ablation" / "all-runtimes",
            **evaluator_kwargs,
        )
        plotter_runtime.generate_runtime_plot(df_results=df_results_configs_only_valid)

    if configs_hyperparameters is not None:
        config_types = {k: v["model_type"] for k, v in configs_hyperparameters.items()}
        plotter_ensemble_weights = TabArenaEvaluator(
            output_dir=eval_save_path / Path("ablation") / "ensemble_weights",
            config_types=config_types,
            **evaluator_kwargs,
        )
        # plotter_ensemble_weights.plot_portfolio_ensemble_weights_barplot(df_ensemble_weights=df_ensemble_weights)
        df_ensemble_weights = plotter_ensemble_weights.get_ensemble_weights(
            df_results=df_results,
            method=portfolio_name,
            aggregate_folds=True,
        )
        plotter_ensemble_weights.plot_portfolio_ensemble_weights_barplot(df_ensemble_weights=df_ensemble_weights)
        plotter_ensemble_weights.plot_ensemble_weights_heatmap(df_ensemble_weights=df_ensemble_weights, figsize=(24, 20))

    if df_results_holdout is not None:
        eval_holdout_ablation(
            df_results=df_results,
            df_results_holdout=df_results_holdout,
            eval_save_path=eval_save_path,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            evaluator_kwargs=evaluator_kwargs,
        )

    if df_results_cpu is not None:
        eval_cpu_vs_gpu_ablation(
            df_results=df_results,
            df_results_cpu=df_results_cpu,
            df_results_configs=df_results_configs,
            eval_save_path=eval_save_path,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            evaluator_kwargs=evaluator_kwargs,
        )

    get_per_dataset_tables(
        df_results=df_results,
        save_path=eval_save_path / "per_dataset"
    )

    use_tabpfn_lst = [False, True]
    use_tabicl_lst = [False, True]
    use_imputation_lst = [False, True]
    problem_type_pst = [None, "cls", "reg", "binary", "multiclass"]
    include_portfolio_lst = [False, True]
    with_baselines_lst = [False, True]
    lite_lst = [False, True]

    all_combinations = list(product(
        use_tabpfn_lst,
        use_tabicl_lst,
        use_imputation_lst,
        problem_type_pst,
        include_portfolio_lst,
        with_baselines_lst,
        lite_lst,
    ))
    n_combinations = len(all_combinations)

    # TODO: Use ray to speed up?
    # plots for sub-benchmarks, with and without imputation
    for i, (use_tabpfn, use_tabicl, use_imputation, problem_type, include_portfolio, with_baselines, lite) in enumerate(all_combinations):
        print(f"Running figure generation {i+1}/{n_combinations}...")

        # combinations to skip
        if problem_type in ["binary", "multiclass"] and (use_tabpfn or use_tabicl or include_portfolio or lite):
            continue

        if not with_baselines and (include_portfolio or lite or use_tabpfn or use_tabicl):
            continue

        folder_name = ("tabpfn-tabicl" if use_tabpfn else "tabicl") \
            if use_tabicl else ("tabpfn" if use_tabpfn else "full")
        baselines = ["AutoGluon 1.3 (4h)"]
        baseline_colors = ["black"]
        if include_portfolio:
            baselines.append("TabArena ensemble (4h)")
            baseline_colors.append("tab:purple")
            folder_name = str(Path("portfolio") / folder_name)
        if lite:
            folder_name = str(Path("lite") / folder_name)
        else:
            folder_name = str(Path("all") / folder_name)
        if use_imputation:
            folder_name = folder_name + "-imputed"
        if not with_baselines:
            baselines = []
            baseline_colors = []
            folder_name = folder_name + "-nobaselines"
        if problem_type is not None:
            folder_name = folder_name + f"-{problem_type}"

        banned_model_types = set()
        imputed_models = []
        if not use_tabicl:
            banned_model_types.add(tabicl_type)
            imputed_models.append("TabICL")
        if not use_tabpfn:
            banned_model_types.add(tabpfn_type)
            imputed_models.append("TabPFNv2")

        datasets = (
            list(set(datasets_tabpfn).intersection(datasets_tabicl)) if use_tabpfn else datasets_tabicl) \
            if use_tabicl else (datasets_tabpfn if use_tabpfn else None)
        if datasets is None:
            datasets = list(task_metadata["name"])

        if problem_type is None:
            problem_types = None
        elif problem_type == "cls":
            problem_types = ["binary", "multiclass"]
        elif problem_type == "reg":
            problem_types = ["regression"]
        elif problem_type == "binary":
            problem_types = ["binary"]
        elif problem_type == "multiclass":
            problem_types = ["multiclass"]
        else:
            raise AssertionError(f"Invalid problem_type value: {problem_type}")

        if use_imputation:
            banned_model_types = set()
        if problem_type == "reg":
            banned_model_types.add(tabicl_type)
        banned_model_types = list(banned_model_types)

        if problem_types:
            datasets = [d for d in datasets if task_metadata[task_metadata["name"] == d].iloc[0]["problem_type"] in problem_types]

        if len(datasets) == 0:
            continue

        plotter = TabArenaEvaluator(
            output_dir=eval_save_path / folder_name,
            datasets=datasets,
            problem_types=problem_types,
            banned_model_types=banned_model_types,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            folds=[0] if lite else None,
            **evaluator_kwargs,
        )

        plotter.eval(
            df_results=df_results,
            baselines=baselines,
            baseline_colors=baseline_colors,
            imputed_names=imputed_models,
            only_datasets_for_method={'TabPFNv2': datasets_tabpfn, 'TabICL': datasets_tabicl},
            plot_extra_barplots=False,
            include_norm_score=not include_portfolio,
            plot_times=True,
            plot_other=False,
        )


def eval_holdout_ablation(
    df_results: pd.DataFrame,
    df_results_holdout: pd.DataFrame,
    eval_save_path: str | Path,
    elo_bootstrap_rounds: int = 100,
    evaluator_kwargs: dict = None,
):
    if evaluator_kwargs is None:
        evaluator_kwargs = {}
    folder_name = Path("ablation") / "holdout"

    df_results = df_results.copy(deep=True)
    df_results_holdout = df_results_holdout.copy(deep=True)

    df_results_holdout["method"] = df_results_holdout["method"].apply(rename_holdout)

    config_types_results = df_results["config_type"].unique()

    config_types_results_holdout = df_results_holdout["config_type"].unique()

    config_types_shared = [c for c in config_types_results if c in config_types_results_holdout]

    # filter to only config_types that are present in both results results_holdout
    df_results = df_results[df_results["config_type"].isna() | df_results["config_type"].isin(config_types_shared)]
    df_results_holdout = df_results_holdout[df_results_holdout["config_type"].isna() | df_results_holdout["config_type"].isin(config_types_shared)]
    df_results = pd.concat([df_results, df_results_holdout], ignore_index=True)

    # only these tune types will be part of the elo plot
    plot_tune_types = ["tuned_ensembled", "holdout_tuned_ensembled"]

    baselines = ["AutoGluon 1.3 (4h)"]
    baseline_colors = ["black"]

    plotter = TabArenaEvaluator(
        output_dir=eval_save_path / folder_name,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        **evaluator_kwargs,
    )

    plotter.eval(
        df_results=df_results,
        baselines=baselines,
        baseline_colors=baseline_colors,
        plot_extra_barplots=True,
        plot_times=False,
        plot_tune_types=plot_tune_types,
        plot_other=False,
    )


def rename_holdout(name: str) -> str:
    if "(default)" in name:
        name = name.replace("(default)", "(holdout)")
    elif "(tuned)" in name:
        name = name.replace("(tuned)", "(tuned, holdout)")
    elif "(tuned + ensemble)" in name:
        name = name.replace("(tuned + ensemble)", "(tuned + ensemble, holdout)")
    return name


def eval_cpu_vs_gpu_ablation(
    df_results: pd.DataFrame,
    df_results_cpu: pd.DataFrame,
    eval_save_path: str | Path,
    df_results_configs: pd.DataFrame = None,
    elo_bootstrap_rounds: int = 100,
    evaluator_kwargs: dict = None,
):
    if evaluator_kwargs is None:
        evaluator_kwargs = {}
    df_results_cpu_gpu = pd.concat([df_results, df_results_cpu], ignore_index=True)

    tabicl_type = "TABICL_GPU"
    tabpfn_type = "TABPFNV2_GPU"

    folder_name = Path("ablation") / "cpu_vs_gpu"

    banned_model_types = [tabpfn_type, tabicl_type]

    baselines = ["AutoGluon 1.3 (4h)"]
    baseline_colors = ["black"]

    plotter = TabArenaEvaluator(
        output_dir=eval_save_path / folder_name,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        banned_model_types=banned_model_types,
        **evaluator_kwargs,
    )

    plotter.eval(
        df_results=df_results_cpu_gpu,
        baselines=baselines,
        baseline_colors=baseline_colors,
        plot_extra_barplots=True,
        plot_times=True,
        plot_other=False,
        only_datasets_for_method={},
    )

    if df_results_configs is not None:
        plotter = TabArenaEvaluator(
            output_dir=eval_save_path / folder_name,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            banned_model_types=banned_model_types,
            **evaluator_kwargs,
        )

        df_results_configs_only_cpu_gpu = df_results_configs[df_results_configs["config_type"].isin([
            "REALMLP",
            "REALMLP_GPU",
            "TABM",
            "TABM_GPU",
            "MNCA",
            "MNCA_GPU",
        ])]

        plotter.generate_runtime_plot(df_results=df_results_configs_only_cpu_gpu)
