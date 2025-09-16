from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import ticker
from tueplots import bundles, fonts, fontsizes
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from autogluon.common.savers import save_pd

from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.tabarena.tabarena import TabArena
from tabrepo.paper.paper_utils import get_framework_type_method_names, get_method_rename_map
from tabrepo.plot.plot_ens_weights import create_heatmap

matplotlib.rcParams.update(fontsizes.neurips2024())
matplotlib.rcParams.update({
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
})


def darken_color(color_str, amount=0.5):
    # Convert color string to RGB tuple (values between 0 and 1)
    rgb = mcolors.to_rgb(color_str)
    # Interpolate with black (0, 0, 0)
    darker_rgb = tuple((1 - amount) * c for c in rgb)
    return darker_rgb


# FIXME: ensemble weights can get if including `config_hyperparameters` as input
class TabArenaEvaluator:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        task_metadata: pd.DataFrame | None = None,
        config_types: dict[str, str] = None,
        method_col: str = "method",
        methods: list[str] | None = None,
        folds: list[int] | None = None,
        datasets: list[str] | None = None,
        problem_types: list[str] | None = None,
        banned_model_types: list[str] | None = None,
        elo_bootstrap_rounds: int = 100,
        keep_best: bool = False,
        figure_file_type: str = "pdf",
        use_latex: bool = False,
    ):
        """

        Parameters
        ----------
        methods
            filter methods
        folds
            filter folds
        datasets
            filter datasets
        problem_types
            filter problem_types
        elo_bootstrap_rounds
            10 = toy
            100 = paper
        kwargs
        """
        if task_metadata is None:
            task_metadata = load_task_metadata()
        self.output_dir = output_dir
        self.task_metadata = task_metadata
        self.method_col = method_col
        self.config_types = config_types
        self.figure_file_type = figure_file_type

        self.datasets = datasets
        self.problem_types = problem_types
        self.methods = methods
        self.folds = folds
        self.elo_bootstrap_rounds = elo_bootstrap_rounds
        self.banned_model_types = banned_model_types
        self.keep_best = keep_best

        self.use_latex = use_latex
        if self.use_latex:
            matplotlib.rcParams.update(bundles.neurips2024())
            matplotlib.rcParams.update(fonts.neurips2024_tex())
            self.rc_context_params = {
                                    'font.family': 'serif',
                                    "text.usetex": True,
                                } | fontsizes.neurips2024(default_smaller=0)
        else:
            self.rc_context_params = {}

    def compute_normalized_error_dynamic(self, df_results: pd.DataFrame) -> pd.DataFrame:
        df_results = df_results.copy(deep=True)
        df_results_og = df_results.copy(deep=True)

        df_results = df_results.drop(columns=["normalized-error-dataset", "normalized-error-task"], errors="ignore")

        method_col = self.method_col

        df_results_per_dataset = df_results.groupby([method_col, "dataset"])["metric_error"].mean().reset_index(
            drop=False)

        # Alternative, this also incorporates Portfolios and HPO into the normalized scoring. This makes normalized-error dependent on what simulations we run.
        # This is unbiased against very strong simulation results because the best method defines what is `0.0` on a dataset.
        normalized_scorer_dataset = NormalizedScorer(
            df_results_per_dataset,
            tasks=list(df_results_per_dataset["dataset"].unique()),
            baseline=None,
            task_col="dataset",
            framework_col=method_col,
        )

        all_tasks = df_results[["dataset", "fold"]].drop_duplicates().values.tolist()
        all_tasks = [tuple(task) for task in all_tasks]

        normalized_scorer_task = NormalizedScorer(
            df_results,
            tasks=all_tasks,
            baseline=None,
            task_col=["dataset", "fold"],
            framework_col=method_col,
        )

        df_results["normalized-error-task"] = [normalized_scorer_task.rank(task=(dataset, fold), error=error) for
                                               (dataset, fold, error) in
                                               zip(df_results["dataset"], df_results["fold"],
                                                   df_results["metric_error"])]

        df_results_per_dataset["normalized-error-dataset"] = [
            normalized_scorer_dataset.rank(task=dataset, error=error) for (dataset, error) in
            zip(df_results_per_dataset["dataset"], df_results_per_dataset["metric_error"])
        ]

        df_results_per_dataset = df_results_per_dataset.set_index(["dataset", method_col], drop=True)[
            "normalized-error-dataset"]
        df_results = df_results.merge(df_results_per_dataset, left_on=["dataset", method_col], right_index=True)

        df_results_og["normalized-error-dataset"] = df_results["normalized-error-dataset"]
        df_results_og["normalized-error-task"] = df_results["normalized-error-task"]
        return df_results_og

    @classmethod
    def _get_config_types(cls, df_results: pd.DataFrame) -> list[str]:
        config_types = sorted([
            config_type for config_type in df_results["config_type"].unique()
            if config_type is not None and isinstance(config_type, str)
        ])
        return config_types

    def eval(
        self,
        df_results: pd.DataFrame,
        include_norm_score: bool = False,
        use_gmean: bool = False,
        imputed_names: list[str] | None = None,
        only_datasets_for_method: dict[str, list[str]] | None = None,
        baselines: list[str] | str | None = "auto",
        baseline_colors: list[str] | None = None,
        plot_tune_types: list[str] | None = None,
        plot_times: bool = False,
        plot_extra_barplots: bool = False,
        plot_cdd: bool = True,
        plot_runtimes: bool = False,
        plot_other: bool = False,
        calibration_framework: str | None = "auto",
    ) -> pd.DataFrame:
        if calibration_framework is not None and calibration_framework == "auto":
            calibration_framework = "RF (default)"
        if baselines is None:
            baselines = []
        elif baselines == "auto":
            baselines = [
                # "AutoGluon 1.3 (5m)",
                # "AutoGluon 1.3 (1h)",
                "AutoGluon 1.3 (4h)",
                # "Portfolio-N200 (ensemble) (4h)",
                # "Portfolio-N200 (ensemble, holdout) (4h)",
            ]
            if baseline_colors is None:
                baseline_colors = [
                    # "darkgray",
                    "black",
                    # "blue",
                    # "red",
                ]
        if baseline_colors is None:
            baseline_colors = []
        assert len(baselines) == len(baseline_colors)
        method_col = self.method_col
        df_results = df_results.copy(deep=True)
        if "seed" not in df_results:
            df_results["seed"] = 0
        if "imputed" not in df_results:
            df_results["imputed"] = False
        df_results["imputed"] = df_results["imputed"].fillna(False).astype(bool)
        df_results["seed"] = df_results["seed"].fillna(0).astype(int)
        df_results = df_results.drop_duplicates(subset=[
            "dataset", "fold", self.method_col, "seed"
        ], keep="first")

        if "normalized-error-dataset" not in df_results:
            df_results = self.compute_normalized_error_dynamic(df_results=df_results)
        assert "normalized-error-dataset" in df_results, f"Run `self.compute_normalized_error_dynamic(df_results)` first to get normalized-error."
        df_results["normalized-error"] = df_results["normalized-error-dataset"]

        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if self.folds is not None:
            df_results = df_results[df_results["fold"].isin(self.folds)]
        if self.methods is not None:
            df_results = df_results[df_results[self.method_col].isin(self.methods)]
        if self.problem_types is not None:
            df_results = df_results[df_results["problem_type"].isin(self.problem_types)]
        if not self.keep_best:
            # FIXME: Don't do regex, use subtype column value
            df_results = df_results[~df_results[self.method_col].str.contains("(best)", regex=False)]

        if self.banned_model_types:
            df_results = df_results[~df_results["config_type"].isin(self.banned_model_types)]
            # framework_types = [f for f in framework_types if f not in self.banned_model_types]
        framework_types = self._get_config_types(df_results=df_results)

        # ----- add times per 1K samples -----
        dataset_to_n_samples_train = self.task_metadata.set_index("name")["n_samples_train_per_fold"].to_dict()
        dataset_to_n_samples_test = self.task_metadata.set_index("name")["n_samples_test_per_fold"].to_dict()

        df_results['time_train_s_per_1K'] = df_results['time_train_s'] * 1000 / df_results["dataset"].map(
            dataset_to_n_samples_train)
        df_results['time_infer_s_per_1K'] = df_results['time_infer_s'] * 1000 / df_results["dataset"].map(
            dataset_to_n_samples_test)

        df_results[self.method_col] = df_results[self.method_col].map({
            "AutoGluon_v130_bq_4h8c": "AutoGluon 1.3 (4h)",

            "RandomForest_r1_BAG_L1_HOLDOUT": "RF (holdout)",
            "ExtraTrees_r1_BAG_L1_HOLDOUT": "XT (holdout)",
            "LinearModel_c1_BAG_L1_HOLDOUT": "LR (holdout)",

            "LightGBM_c1_BAG_L1_HOLDOUT": "GBM (holdout)",
            "XGBoost_c1_BAG_L1_HOLDOUT": "XGB (holdout)",
            "CatBoost_c1_BAG_L1_HOLDOUT": "CAT (holdout)",
            "NeuralNetTorch_c1_BAG_L1_HOLDOUT": "NN_TORCH (holdout)",
            "NeuralNetFastAI_c1_BAG_L1_HOLDOUT": "FASTAI (holdout)",

            "RealMLP_c1_BAG_L1_HOLDOUT": "REALMLP (holdout)",
            "ExplainableBM_c1_BAG_L1_HOLDOUT": "EBM (holdout)",
            "TabM_c1_BAG_L1_HOLDOUT": "TABM (holdout)",
            "ModernNCA_c1_BAG_L1_HOLDOUT": "MNCA (holdout)",

        }).fillna(df_results[self.method_col])
        # print(df_results)

        df_results_rank_compare = copy.deepcopy(df_results)
        df_results_unfiltered = copy.deepcopy(df_results)

        # FIXME: (Nick) Unsure which form of the df should go in here?
        # David H: doesn't matter since results are not relative to other methods in the df
        if only_datasets_for_method is not None and plot_times:
            self.plot_tabarena_times(df=df_results_unfiltered, output_dir=self.output_dir,
                                     only_datasets_for_method=only_datasets_for_method, show=False)

        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        # also remove portfolio baselines except AutoGluon?
        df_results_rank_compare = df_results_rank_compare[(~df_results_rank_compare[self.method_col].map(f_map_type).isna()) | (df_results_rank_compare[self.method_col].isin(baselines))]

        # recompute normalized errors
        df_results_rank_compare = self.compute_normalized_error_dynamic(df_results_rank_compare)

        # ----- end removing unused methods -----

        hue_order_family_proportion = [
            "CatBoost",
            "TabPFNv2",
            "TabM",
            "ModernNCA",
            "TabDPT",
            "LightGBM",
            "TabICL",

            "RandomForest",
            "XGBoost",
            "RealMLP",

            "NeuralNetFastAI",
            "ExplainableBM",
            "NeuralNetTorch",
            "ExtraTrees",

            # "LinearModel",
            # "KNeighbors",

            # "TabForestPFN",

            # "FTTransformer",

        ]

        # FIXME: TODO (Nick): Move this to its own class for utility plots, no need to re-plot this in every eval call.
        # plot_family_proportion(df=df_results_unfiltered, save_prefix=f"{self.output_dir}/figures/family_prop",
        #                        method="Portfolio-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion,
        #                        show=False)

        if include_norm_score:
            self.plot_tuning_impact(
                df=df_results_rank_compare,
                framework_types=framework_types,
                save_prefix=f"{self.output_dir}",
                use_gmean=use_gmean,
                baselines=baselines,
                baseline_colors=baseline_colors,
                use_score=True,
                name_suffix="-normscore-dataset-horizontal",
                imputed_names=imputed_names,
                plot_tune_types=plot_tune_types,
                show=False,
                use_y=True
            )

            if plot_extra_barplots:
                self.plot_tuning_impact(
                    df=df_results_rank_compare,
                    framework_types=framework_types,
                    save_prefix=f"{self.output_dir}",
                    use_gmean=use_gmean,
                    baselines=baselines,
                    baseline_colors=baseline_colors,
                    use_score=True,
                    name_suffix="-normscore-dataset",
                    imputed_names=imputed_names,
                    plot_tune_types=plot_tune_types,
                    show=False,
                )

                self.plot_tuning_impact(
                    df=df_results_rank_compare,
                    framework_types=framework_types,
                    save_prefix=f"{self.output_dir}",
                    use_gmean=use_gmean,
                    baselines=baselines,
                    baseline_colors=baseline_colors,
                    use_score=True,
                    metric="normalized-error-task",
                    name_suffix="-normscore-task",
                    imputed_names=imputed_names,
                    plot_tune_types=plot_tune_types,
                    show=False,
                )

        tabarena = TabArena(
            method_col=method_col,
            task_col="dataset",
            seed_column="fold",
            error_col="metric_error",
            columns_to_agg_extra=[
                "time_train_s",
                "time_infer_s",
                "time_train_s_per_1K",
                "time_infer_s_per_1K",
                "normalized-error",
                "normalized-error-task",
                "imputed",
            ],
            groupby_columns=[
                "metric",
                "problem_type",
            ],
        )

        leaderboard = tabarena.leaderboard(
            data=df_results_rank_compare,
            include_winrate=True,
            include_mrr=True,
            include_rank_counts=True,
            include_elo=True,
            elo_kwargs=dict(
                calibration_framework=calibration_framework,
                calibration_elo=1000,
                BOOTSTRAP_ROUNDS=self.elo_bootstrap_rounds,
            )
        )
        elo_map = leaderboard["elo"]
        leaderboard = leaderboard.reset_index(drop=False)
        save_pd.save(path=f"{self.output_dir}/tabarena_leaderboard.csv", df=leaderboard)

        self.create_leaderboard_latex(leaderboard, framework_types=framework_types, save_dir=self.output_dir)

        print(
            f"Evaluating with {len(df_results_rank_compare[tabarena.task_col].unique())} datasets... | problem_types={self.problem_types}, folds={self.folds}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard)

        # horizontal elo barplot
        self.plot_tuning_impact(
            df=df_results_rank_compare,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo-horizontal",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            use_y=True,
            show=False
        )

        # vertical elo barplot
        self.plot_tuning_impact(
            df=df_results_rank_compare,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            show=False
        )

        results_per_task = tabarena.compute_results_per_task(data=df_results_rank_compare)

        def rename_model(name: str):
            parts = name.split(" ")
            if parts[0] in f_map_type_name:
                parts[0] = f_map_type_name[parts[0]]
            name = " ".join(parts)
            return name.replace('(tuned + ensemble)', '(tuned + ensembled)')

        # use tuned+ensembled version if available, and default otherwise
        tune_methods = results_per_task[self.method_col].map(f_map_inverse)
        method_types = results_per_task[self.method_col].map(f_map_type).fillna(results_per_task[self.method_col])
        tuned_ens_types = method_types[tune_methods == 'tuned_ensembled']
        results_te_per_task = results_per_task[(tune_methods == 'tuned_ensembled') | ((tune_methods == 'default') & ~method_types.isin(tuned_ens_types))]

        # rename model part
        results_te_per_task.loc[:, self.method_col] = results_te_per_task[self.method_col].map(rename_model)

        if plot_cdd:
            tabarena.plot_critical_diagrams(
                results_per_task=results_te_per_task,
                save_path=f"{self.output_dir}/figures/critical-diagram.{self.figure_file_type}",
                show=False,
            )

        if plot_runtimes:
            self.generate_runtime_plot(df_results=df_results_rank_compare)

        if plot_other:
            try:
                import autogluon_benchmark
            except:
                print(f"WARNING: autogluon_benchmark failed to import... skipping extra figure generation")
            else:
                results_per_task_ag_benchmark = results_per_task.rename(columns={
                    "champ_delta": "bestdiff",
                })
                self.run_autogluon_benchmark_logic(
                    results_per_task=results_per_task_ag_benchmark,
                    elo_map=elo_map,
                    tabarena=tabarena,
                    calibration_framework=calibration_framework,
                )

        return leaderboard

    def run_autogluon_benchmark_logic(self, results_per_task: pd.DataFrame, elo_map: dict, tabarena: TabArena,
                                      calibration_framework: str):
        """
        Requires autogluon_benchmark installed:

        Parameters
        ----------
        results_per_task
        elo_map
        tabarena
        calibration_framework

        Returns
        -------

        """
        results_per_task_rename = results_per_task.rename(columns={
            tabarena.method_col: "framework",
            tabarena.task_col: "dataset",
            tabarena.error_col: "metric_error",
        })

        results_per_task_rename["elo"] = results_per_task_rename["framework"].map(elo_map)

        # Nick: Comment out this code if you don't have autogluon_benchmark
        from autogluon_benchmark.plotting.plotter import Plotter
        plotter = Plotter(
            results_ranked_df=results_per_task_rename,
            results_ranked_fillna_df=results_per_task_rename,
            save_dir=f"{self.output_dir}/figures/plotter"
        )

        # FIXME: Nick: This isn't yet merged, as I haven't made it nice yet
        # plotter.plot_pareto_time_infer_elo(data=results_per_task_rename)
        # plotter.plot_pareto_time_train_elo(data=results_per_task_rename)

        plotter.plot_all(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=self.elo_bootstrap_rounds,
        )

    def get_method_rename_map(self) -> dict[str, str]:
        return get_method_rename_map()  # FIXME: Avoid hardcoding

    def plot_portfolio_ensemble_weights_barplot(self, df_ensemble_weights: pd.DataFrame):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pathlib import Path
        import matplotlib.colors as mcolors
        import numpy as np

        fig, ax = plt.subplots(1, 1,
                               figsize=(3.5, 3)
                               )

        df_ensemble_weights = df_ensemble_weights.copy(deep=True)
        _method_rename_map = self.get_method_rename_map()
        columns_new = [_method_rename_map.get(c, c) for c in df_ensemble_weights.columns]
        df_ensemble_weights.columns = columns_new

        df_long = df_ensemble_weights.melt(var_name="Model", value_name="Weight")
        model_order = list(df_ensemble_weights.columns)

        pastel_palette = sns.color_palette("pastel")
        deep_palette = sns.color_palette("deep")

        # Define gradient from pastel and deep separately
        pastel_start = mcolors.to_rgb(pastel_palette[2])
        pastel_end = mcolors.to_rgb(pastel_palette[0])
        deep_start = mcolors.to_rgb(deep_palette[2])
        deep_end = mcolors.to_rgb(deep_palette[0])

        # Create pastel gradient for bars
        bar_colors = [mcolors.to_hex(c) for c in np.linspace(pastel_start, pastel_end, len(model_order))]

        # Create deep gradient for error bars
        error_colors = [mcolors.to_hex(c) for c in np.linspace(deep_start, deep_end, len(model_order))]

        # Alphas for bars
        alphas = np.linspace(1.0, 1.0, len(model_order))[::-1]  # Keep at 1.0 for now

        # Create barplot
        barplot = sns.barplot(
            data=df_long,
            x="Weight",
            y="Model",
            hue="Model",
            legend=False,
            ax=ax,
            order=model_order,
            palette=bar_colors,
        )

        # Apply alpha to bar colors
        for patch, alpha in zip(barplot.patches, alphas):
            r, g, b = patch.get_facecolor()[:3]
            patch.set_facecolor((r, g, b, alpha))

        # Update error bar colors manually
        for i, line in enumerate(ax.lines):
            # Seaborn/matplotlib adds error bar lines in a certain order.
            # Each bar usually has 2 lines: one vertical bar and one cap on top.
            # Here, we assume two lines per error bar, so divide i by 2.
            color_index = i // 2
            if color_index < len(error_colors):
                line.set_color(error_colors[color_index])

        barplot.set_xlabel("Average weight in TabArena ensemble")
        barplot.set_ylabel("")

        fig_name = f"portfolio-weight-barplot.{self.figure_file_type}"
        fig_prefix = Path(self.output_dir) / "figures"
        fig_prefix.mkdir(parents=True, exist_ok=True)

        fig_save_path = fig_prefix / fig_name
        plt.savefig(fig_save_path)

    def create_leaderboard_latex(self, df: pd.DataFrame, framework_types, save_dir):
        df = df.copy(deep=True)
        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        def rename_model(name: str):
            parts = name.split(" ")
            if parts[0] in f_map_type_name:
                parts[0] = f_map_type_name[parts[0]]
            name = " ".join(parts)
            name = name.replace('(default)', '(D)')
            name = name.replace('(tuned)', '(T)')
            name = name.replace('(tuned + ensemble)', '(T+E)')
            return name

        df = df.sort_values(by="elo", ascending=False)

        df_new = pd.DataFrame()

        df_new[r"Model"] = df[self.method_col].map(rename_model)
        # do the more annoying way {}_{...} so that \textbf{} affects the main number
        df_new[r"Elo ($\uparrow$)"] = [f'{round(elo)}' + r'${}_{' + f'-{math.ceil(elom)},+{math.ceil(elop)}' + r'}$'
                                       for elo, elom, elop in zip(df["elo"], df["elo-"], df["elo+"])]
        df_new[r"Norm." + "\n" + r"score ($\uparrow$)"] = [f'{1-err:5.3f}' for err in df["normalized-error"]]
        df_new[r"Avg." + "\n" + r"rank ($\downarrow$)"] = [f'{rank:.1f}' for rank in df["rank"]]
        df_new["Harm.\nmean\n" + r"rank ($\downarrow$)"] = [f'{1/val:.1f}' for val in df["mrr"]]
        df_new[r"\#wins ($\uparrow$)"] = [str(cnt) for cnt in df["rank=1_count"]]
        df_new[f"Improva-\n" + r"bility ($\downarrow$)"] = [f'{100*val:.1f}\\%' for val in df["champ_delta"]]
        df_new[r"Train time" + "\n" + r"per 1K [s]"] = [f'{t:.2f}' for t in df["median_time_train_s_per_1K"]]
        df_new[r"Predict time" + "\n" + r"per 1K [s]"] = [f'{t:.2f}' for t in df["median_time_infer_s_per_1K"]]

        # ----- highlight best and second-best numbers per column -----

        # first, convert the strings back to floats
        def extract_first_float(s):
            """
            Extracts the first sequence of digits (including decimal point) from the input string
            and returns it as a float. Returns None if no valid number is found.
            """
            match = re.search(r'\d+(\.\d+)?', s)
            if match:
                return float(match.group())
            return None

        def find_smallest_and_second_smallest_indices(numbers):
            if len(numbers) < 2:
                return [], []

            # Find the smallest value
            min_val = min(numbers)
            min_indices = [i for i, x in enumerate(numbers) if x == min_val]

            # Exclude the smallest values and find the second smallest
            remaining = [x for x in numbers if x != min_val]
            if not remaining:
                return min_indices, []  # No second smallest

            second_min_val = min(remaining)
            second_min_indices = [i for i, x in enumerate(numbers) if x == second_min_val]

            return min_indices, second_min_indices

        # then, add textbf or underline to the correct rows
        for col_idx, col in enumerate(df_new.columns):
            if r'\uparrow' in col or r'\downarrow' in col:
                # factor = 1 if r'\downarrow' in col else -1
                # numbers = [factor * extract_first_float(s) for s in df_new[col]]
                ranks = df_new[col].map(extract_first_float).rank(method="min", ascending=r'\downarrow' in col)
                for rank, color in [(1, 'gold'), (2, 'silver'), (3, 'bronze')]:
                    df_new.loc[ranks == rank, col] = df_new.loc[ranks == rank, col].apply(
                        lambda x: f"\\textcolor{{{color}}}{{\\textbf{{{x}}}}}"
                    )

                # min_indices, second_min_indices = find_smallest_and_second_smallest_indices(numbers)
                # for idx in min_indices:
                #     df_new.iloc[idx, col_idx] = r'\textbf{' + df_new.iloc[idx, col_idx] + r'}'
                # for idx in second_min_indices:
                #     df_new.iloc[idx, col_idx] = r'\underline{' + df_new.iloc[idx, col_idx] + r'}'


        # ----- create latex table -----

        rows = []
        rows.append(r'\begin{tabular}{' + 'llcccccrr' + r'}')
        rows.append(r'\toprule')
        # rows.append(' & '.join(df_new.columns) + r' \\')

        col_names_split = [col.split('\n') for col in df_new.columns]
        n_rows_header = max([len(rows) for rows in col_names_split])
        for row_idx in range(n_rows_header):
            rows.append(' & '.join([r'\textbf{' + lst[row_idx] + r'}' if row_idx < len(lst) else ''
                                    for lst in col_names_split]) + r' \\')
        rows.append(r'\midrule')

        for row_index, row in df_new.iterrows():
            rows.append(' & '.join([row[col_name] for col_name in df_new.columns]) + r' \\')

        rows.append(r'\bottomrule')
        rows.append(r'\end{tabular}')

        table = '\n'.join(rows)

        with open(Path(save_dir) / 'leaderboard.tex', 'w') as f:
            f.write(table)

    # FIXME: Avoid hardcoding
    def plot_tuning_impact(
        self,
        df: pd.DataFrame,
        framework_types: list,
        save_prefix: str,
        baselines: list[str] = None,
        baseline_colors: list[str] = None,
        show: bool = True,
        use_gmean=False,
        use_score: bool = True,
        df_elo: pd.DataFrame = None,
        name_suffix: str | None = None,
        imputed_names: list[str] | None = None,
        use_y: bool = False,
        metric: str = "normalized-error",
        plot_tune_types: list[str] | None = None,
    ):
        same_width = use_y
        use_lim = True
        use_elo = df_elo is not None
        lower_is_better = True
        lim = None
        xlim = None
        ylim = None

        if imputed_names is None:
            imputed_names = []
        # imputed_names = imputed_names or ['TabPFNv2', 'TabICL']

        df = df.copy(deep=True)

        framework_col = "framework_type"
        # framework_col = "framework_name"

        groupby_columns_extra = ["dataset"]

        if use_elo:
            metric = "elo"
            use_lim = True
            lim = [500, None]
            lower_is_better = False
            df = df_elo.copy(deep=True)
            df = df[[self.method_col, "elo", "elo+", "elo-"]]
            groupby_columns_extra = []
        elif use_score:
            lower_is_better = False
            df["normalized-score"] = 1 - df[metric]
            # df_plot_w_mean_per_dataset["normalized-score"] = 1 - df_plot_w_mean_per_dataset["normalized-error"]
            metric = "normalized-score"
        else:
            metric = metric

        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        df["framework_type"] = df[self.method_col].map(f_map_type).fillna(df[self.method_col])
        df["tune_method"] = df[self.method_col].map(f_map_inverse).fillna("default")

        if baselines is None:
            baselines = []
        if baseline_colors is not None:
            assert len(baselines) == len(
                baseline_colors), f"A color must be specified for each baseline via the `baseline_colors` argument."

        framework_types = baselines + framework_types

        df["framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])
        framework_types = [f_map_type_name[ft] if ft in f_map_type_name else ft for ft in framework_types]

        if plot_tune_types:
            df = df[df["tune_method"].isin(plot_tune_types) | df[self.method_col].isin(baselines)]

        df_plot = df[df["framework_type"].isin(framework_types)]
        # df_plot = df_plot[~df_plot["framework_type"].isin(imputed_names)]

        # pd.set_option('display.max_columns', None)  # todo
        # print(f'{df_plot.head()=}')

        # df_plot_w_mean_2 = df_plot.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()

        df_plot_w_mean_per_dataset = df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[
            metric].mean().reset_index()

        if use_gmean:
            # FIXME: Doesn't plot correctly, need to figure out error bars for geometric mean
            df_plot_eps = df_plot.copy(deep=True)
            df_plot_eps[metric] += 0.01
            from scipy.stats import gmean
            df_plot_w_gmean_per_dataset = df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[
                metric].apply(gmean).reset_index()
            df_plot_w_mean_per_dataset = df_plot_w_gmean_per_dataset

        df_plot_w_mean_2 = df_plot_w_mean_per_dataset.groupby(["framework_type", "tune_method"])[
            metric].mean().reset_index()

        df_plot_w_mean_2 = df_plot_w_mean_2.sort_values(by=metric, ascending=lower_is_better)
        baseline_means = {}
        for baseline in baselines:
            baseline_means[baseline] = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] == baseline][metric].iloc[0]

        df_plot_w_mean_2 = df_plot_w_mean_2[~df_plot_w_mean_2["framework_type"].isin(baselines)]

        df_plot_mean_dedupe = df_plot_w_mean_2.drop_duplicates(subset=["framework_type"], keep="first")

        framework_type_order = list(df_plot_mean_dedupe["framework_type"].to_list())
        framework_type_order.reverse()

        # change to names
        # df_plot_w_mean_per_dataset["framework_type"] = df_plot_w_mean_per_dataset["framework_type"].map(f_map_type_name)

        # sns.set_color_codes("pastel")
        # with sns.plotting_context("notebook", font_scale=0.8, rc={
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'font.size': 10.95,
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        #     # 'legend.framealpha': 0.5,
        #     'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
        # }):

        with sns.axes_style("whitegrid"):
            # with plt.rc_context({'font.family': 'serif', "text.usetex": True, 'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 12}):
            with plt.rc_context(self.rc_context_params
                                # | figsizes.neurips2024(height_to_width_ratio=0.8)
                                ):
            # with plt.rc_context(fontsizes.neurips2024() | fonts.neurips2024()):
                # with plt.rc_context(figsizes.neurips2024(height_to_width_ratio=0.8)):
                colors = sns.color_palette("pastel").as_hex()
                errcolors = sns.color_palette("deep").as_hex()

                if use_lim and not lim:
                    lim = [0, None]
                if use_y:
                    pos = metric
                    y = framework_col
                    figsize = (3.5, 3)
                    xlim = lim

                    framework_type_order.reverse()

                else:
                    pos = framework_col
                    y = metric
                    ylim = lim
                    figsize = (7, 2.7)
                    # figsize = None

                fig, ax = plt.subplots(1, 1, figsize=figsize)
                # fig, ax = plt.subplots(1, 1)

                if use_y:
                    baseline_func = ax.axvline
                else:
                    baseline_func = ax.axhline

                linewidth = 0.0 if use_y else 0.3
                err_linewidth = 1.6
                err_linewidths = {
                    'tuned_ensembled': err_linewidth,
                    'tuned': err_linewidth * 0.8,
                    'default': err_linewidth * 0.6,
                    'holdout_tuned_ensembled': err_linewidth * 0.6,
                }
                err_alpha = 0.6

                to_plot = [
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned + Ensembled",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled"],
                        ax=ax,
                        order=framework_type_order, color=colors[2],
                        width=0.6, linewidth=linewidth,
                        err_kws={"color": errcolors[2], "linewidth": err_linewidths['tuned_ensembled'], 'alpha': err_alpha},
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Default (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[4],
                    #     width=0.7, linewidth=linewidth,
                    #     err_kws={"color": errcolors[4]},
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[5],
                    #     width=0.65, linewidth=linewidth,
                    #     err_kws={"color": errcolors[5]},
                    # ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned"], ax=ax,
                        order=framework_type_order,
                        color=colors[1],
                        width=0.5, linewidth=linewidth,
                        err_kws={"color": errcolors[1], "linewidth": err_linewidths['tuned'], 'alpha': err_alpha},
                    ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Default",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "default"], ax=ax,
                        order=framework_type_order, color=colors[0],
                        width=0.4, linewidth=linewidth,
                        err_kws={"color": errcolors[0], "linewidth": err_linewidths['default'], 'alpha': err_alpha},
                        alpha=1.0,
                    ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned + Ensembled (Holdout)",
                        data=df_plot_w_mean_per_dataset[
                            df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned_ensembled"], ax=ax,
                        order=framework_type_order,
                        color=colors[3],
                        width=0.3, linewidth=linewidth,
                        err_kws={"color": errcolors[3], "linewidth": err_linewidths['holdout_tuned_ensembled'],
                                 'alpha': err_alpha},
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Best",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "best"], ax=ax,
                    #     order=framework_type_order, color=colors[3],
                    #     width=0.55, linewidth=linewidth,
                    #     err_kws={"color": errcolors[3]},
                    #     alpha=1.0,
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned (4h)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_4h"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[4],
                    #     width=0.5, linewidth=linewidth,
                    #     err_kws={"color": errcolors[4]},
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned + Ensembled (4h)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled_4h"], ax=ax,
                    #     order=framework_type_order, color=colors[5],
                    #     width=0.4,
                    #     err_kws={"color": errcolors[5]},
                    # ),

                ]

                if use_score:
                    widths = [plot_line["width"] for plot_line in to_plot]
                    colors = [plot_line["color"] for plot_line in to_plot]
                    err_kws_lst = [plot_line["err_kws"] for plot_line in to_plot]

                    # to_plot.reverse()
                    for plot_line, width, color, err_kws in zip(to_plot, widths, colors, err_kws_lst):
                        if same_width:
                            plot_line["width"] = 0.6 * 1.3
                        else:
                            plot_line["width"] = width * 1.3
                        # plot_line["color"] = color
                        # plot_line["err_kws"] = err_kws

                for plot_line in to_plot:
                    boxplot = sns.barplot(**plot_line)

                if use_y:
                    boxplot.set(xlabel='Elo' if metric=='elo' else 'Normalized score', ylabel=None)
                else:
                    boxplot.set(xlabel=None, ylabel='Elo' if metric=='elo' else 'Normalized score')  # remove method in the x-axis
                # boxplot.set_title("Effect of tuning and ensembling")

                # # FIXME: (Nick) HACK, otherwise it isn't in the plot, don't know why
                # if use_elo:
                #     if baseline_means and "Portfolio-N200 (ensemble) (4h)" in baselines:
                #         max_baseline_mean = max([v for k, v in baseline_means.items()])
                #         if ylim is not None:
                #             ylim[1] = max_baseline_mean + 50
                #         if xlim is not None:
                #             xlim[1] = max_baseline_mean + 50

                # do this before setting x/y limits
                for baseline_idx, (baseline, color) in enumerate(zip(baselines, baseline_colors)):
                    baseline_mean = baseline_means[baseline]
                    # baseline_func(baseline_mean, label=baseline, color=color, linewidth=2.0, ls="--")
                    baseline_func(baseline_mean, color=color, linewidth=2.0, ls="--", zorder=-10)

                    if baseline == 'Portfolio-N200 (ensemble) (4h)':
                        baseline = 'TabArena ensemble (4h)'

                    if use_y:
                        ax.text(y=(1 - 0.035 * (1 + 2*(len(baselines) - 1 - baseline_idx))) * ax.get_ylim()[0],
                                x=baseline_mean * 0.985, s=baseline, ha='right', color=color)
                    else:
                        ax.text(x=0.5, y=baseline_mean * 0.97, s=baseline, va='top', color=color)


                if ylim is not None:
                    ax.set_ylim(ylim)
                if xlim is not None:
                    ax.set_xlim(xlim)

                ticks = boxplot.get_yticks() if use_y else boxplot.get_xticks()
                ticklabels = [tick.get_text() for tick in
                               (boxplot.get_yticklabels() if use_y else boxplot.get_xticklabels())]

                if use_elo:
                    # ----- add elo error bars -----
                    # Get the bar positions


                    # Add asymmetric error bars manually
                    for pos, framework_type in zip(ticks, ticklabels):
                        for tune_method, errcolor in zip(["default", "tuned", "tuned_ensembled", "holdout_tuned_ensembled"], errcolors):
                            row = df_plot.loc[(df_plot["framework_type"] == framework_type) & (df_plot["tune_method"] == tune_method)]
                            if len(row) == 1:
                                # not all methods have tuned or tuned_ensembled
                                y = row['elo'].values
                                yerr_low = row['elo-'].values
                                yerr_high = row['elo+'].values
                                if use_y:
                                    plotline, caps, barlinecols = plt.errorbar(y, pos, xerr=[yerr_low, yerr_high],
                                                                               fmt='none', color=errcolor,
                                                                               alpha=err_alpha,
                                                                               linewidth=err_linewidths[tune_method])
                                else:
                                    plotline, caps, barlinecols = plt.errorbar(pos, y, yerr=[yerr_low, yerr_high],
                                                                               fmt='none', color=errcolor,
                                                                               alpha=err_alpha,
                                                                               linewidth=err_linewidths[tune_method])
                                # don't round because it will make the lines longer
                                # plt.setp(barlinecols[0], capstyle="round")


                # ----- highlight bars that contain imputed results -----

                # Map x-tick positions to category labels
                label_lookup = dict(zip(ticks, ticklabels))

                has_imputed = False

                for i, bar in enumerate(boxplot.patches):
                    # Get position and convert to category label
                    pos = bar.get_y() + bar.get_height() / 2 if use_y else bar.get_x() + bar.get_width() / 2
                    category_index = round(pos)  # x-ticks are usually 0, 1, 2, ...
                    category = label_lookup.get(category_index)

                    if category in imputed_names:
                        has_imputed = True
                        bar.set_hatch('xx')

                if not use_y:
                    # ----- alternate rows of x tick labels -----
                    # Get current x tick labels
                    labels = [label.get_text() for label in boxplot.get_xticklabels()]

                    # Add newline to every second label
                    new_labels = [label if i % 2 == 0 else r'$\uparrow$' + '\n' + label for i, label in enumerate(labels)]

                    # Apply modified labels
                    boxplot.set_xticks(labels)
                    boxplot.set_xticklabels(new_labels)

                # remove unnecessary extra space on the sides
                if use_y:
                    plt.ylim(len(boxplot.get_yticklabels()) - 0.35, -0.65)
                else:
                    plt.xlim(-0.5, len(boxplot.get_xticklabels()) - 0.5)


                # ax.legend(loc="upper center", ncol=5)
                # these are not the final legend parameters, see below
                ax.legend(loc="upper center", bbox_to_anchor=[0.5, 1.02])

                # reordering the labels
                handles, labels = ax.get_legend_handles_labels()

                # this doesn't work, it also removes the hatch from the actual bars in the plot
                # for handle in handles:
                #     patches = []
                #     if isinstance(handle, Patch):
                #         patches = [handle]
                #     elif isinstance(handle, BarContainer):
                #         patches = handle.patches
                #     for patch in patches:
                #         # remove hatch from existing handles
                #         # It can be present if one of the imputed methods is the best method, e.g., for multiclass
                #         patch.set(hatch=None)

                if has_imputed:
                    # Create a custom legend patch for "imputed"
                    imputed_patch = Patch(facecolor='gray', edgecolor='white', hatch='xx', label='Partially imputed')

                    # Add to existing legend
                    handles.append(imputed_patch)
                    labels.append("Partially imputed")

                # quick fix
                is_holdout_plot = "Tuned + Ensembled (Holdout)" in labels
                if is_holdout_plot:
                    valid_idxs = [i for i, label in enumerate(labels) if label != "Default"]
                    labels = [labels[i] for i in valid_idxs]
                    handles = [handles[i] for i in valid_idxs]

                # specify order
                # len_baselines = len(baselines)
                # len_baselines = 0  # we don't put them in the legend anymore
                # num_other = len(labels) - len_baselines
                # order = [n + len_baselines for n in range(num_other)] + [n for n in range(len_baselines)]
                # order = [3, 4, 5, 0, 1, 2]
                order = list(range(len(labels)))
                order = list(reversed(order))
                # if len(order) == 3:
                #     order = [2, 1, 0]

                # pass handle & labels lists along with order as below
                ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
                          ncol=(len(labels)+1)//2 if has_imputed and use_y else len(labels),
                          bbox_to_anchor=[0.35 if use_y else 0.5, 1.05])

                # if use_y:
                #     boxplot.margins(y=0.05)
                # else:
                #     boxplot.margins(x=0.05)

                # ax.legend(bbox_to_anchor=[0.1, 0.5], loc='center left', ncol=5)
                plt.tight_layout()

                if save_prefix:
                    if name_suffix is None:
                        name_suffix = ""
                    fig_path = Path(save_prefix)
                    fig_path.mkdir(parents=True, exist_ok=True)
                    if use_gmean:
                        fig_name = f"tuning-impact-gmean{name_suffix}.{self.figure_file_type}"
                    else:
                        fig_name = f"tuning-impact{name_suffix}.{self.figure_file_type}"
                    fig_save_path = fig_path / fig_name
                    plt.savefig(fig_save_path)
                if show:
                    plt.show()

    def plot_tabarena_times(self, df: pd.DataFrame, output_dir: str,
                            only_datasets_for_method: dict[str, list[str]] | None = None, show: bool = True):
        # filter to only common datasets
        if only_datasets_for_method is not None:
            for datasets in only_datasets_for_method.values():
                df = df[df["dataset"].isin(datasets)]

        framework_types = self._get_config_types(df_results=df)

        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        df["framework_type"] = df[self.method_col].map(f_map_type).fillna(df[self.method_col])
        df["tune_method"] = df[self.method_col].map(f_map_inverse).fillna("default")
        df = df[df["tune_method"].isin(["default", "tuned_ensembled"])]
        df = df[df['framework_type'].isin(framework_types)]
        df["framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])

        gpu_methods = ['TabICL', 'TabDPT', 'TabPFNv2', "ModernNCA", "TabM"]

        if only_datasets_for_method is not None:
            for method, datasets in only_datasets_for_method.items():
                mask = (df['framework_type'] == method) & (~df['dataset'].isin(datasets))
                # print(f"{df[mask]=}")
                df.loc[mask, 'time_train_s_per_1K'] = np.nan
                df.loc[mask, 'time_infer_s_per_1K'] = np.nan
                # print(f"{df[mask]['time_train_s_per_1K']=}")
                # print(f"{df[mask]['time_infer_s_per_1K']=}")

        # add device name
        framework_types = df["framework_type"].unique()
        device_map = {
            ft: f'{ft} ' + r'(GPU)' if ft in gpu_methods else f'{ft} (CPU)' if not ft.endswith("(CPU)") else ft for ft in framework_types
        }
        device_map = {}
        for ft in framework_types:
            if ft in gpu_methods:
                ft_new = f'{ft} (GPU)'
            elif ft.endswith("(CPU)"):
                ft_new = ft
            elif ft.endswith("(GPU)"):
                ft_new = ft
            else:
                ft_new = f'{ft} (CPU)'
            device_map[ft] = ft_new

        df["framework_type"] = df["framework_type"].map(device_map).fillna(df["framework_type"])

        # take mean times
        df = df.groupby(['dataset', 'framework_type', 'tune_method'])[['time_train_s_per_1K', 'time_infer_s_per_1K']].mean().reset_index()
        df = df.groupby(['framework_type', 'tune_method'])[['time_train_s_per_1K', 'time_infer_s_per_1K']].median().reset_index()

        # ----- ChatGPT plotting code -----

        # Unique values for mapping
        # Sort frameworks by max train time
        sorted_frameworks = (
            df.groupby('framework_type')['time_train_s_per_1K']
            .min()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        frameworks = sorted_frameworks
        y_positions = np.arange(len(frameworks))

        # Maps for tuning method to color and marker
        tune_methods = df['tune_method'].unique()
        # color_map = {tm: c for tm, c in zip(tune_methods, plt.cm.tab10.colors)}
        sns_colors = sns.color_palette("muted").as_hex()
        # sns_colors = sns.color_palette("pastel").as_hex()
        color_map = {'default': sns_colors[0], 'tuned': sns_colors[1], 'tuned_ensembled': sns_colors[2]}
        marker_list = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']
        marker_map = {tm: m for tm, m in zip(tune_methods, marker_list)}

        # Create side-by-side subplots with shared y-axis
        fig, (ax_train, ax_infer) = plt.subplots(
            1, 2, sharey=True, figsize=(5, 4)
        )

        # Alternate row background on both axes
        for i in range(0, len(frameworks), 2):
            for ax in [ax_train, ax_infer]:
                ax.axhspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.3, zorder=0)

        # Plot training and inference times
        for i, fw in enumerate(frameworks):
            df_fw = df[df['framework_type'] == fw]
            for _, row in df_fw.iterrows():
                color = color_map[row['tune_method']]
                marker = marker_map[row['tune_method']]
                ax_train.plot(row['time_train_s_per_1K'], i, marker=marker, color=color, linestyle='None')
                ax_infer.plot(row['time_infer_s_per_1K'], i, marker=marker, color=color, linestyle='None')

        # Train time axis
        ax_train.set_xscale('log')
        ax_train.set_xlabel("Median time per 1K samples [s]")
        ax_train.set_title(r"\textbf{Train+val time}", fontweight='bold')
        ax_train.set_yticks(y_positions)
        ax_train.set_yticklabels(frameworks, fontsize=10)
        ax_train.grid(True, axis='x', alpha=0.5)

        # Inference time axis
        ax_infer.set_xscale('log')
        ax_infer.set_xlabel("Median time per 1K samples [s]")
        ax_infer.set_title(r"\textbf{Inference time}", fontweight='bold')
        ax_infer.set_yticks(y_positions)
        ax_infer.tick_params(labelleft=False)  # Explicitly hide y-tick labels
        ax_infer.grid(True, axis='x', alpha=0.5)

        for ax in [ax_train, ax_infer]:
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
            # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            # ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        tune_method_display_names = {
            'default': 'Default',
            'tuned': 'Tuned',
            'tuned_ensembled': 'Tuned + Ensembled'
        }

        # Add legend above both plots
        legend_elements = [
            plt.Line2D([0], [0], marker=marker_map[tm], color=color_map[tm],
                       linestyle='None', label=tune_method_display_names[tm], markersize=8)
            for tm in tune_methods
        ]
        fig.legend(handles=legend_elements,  # title='Tuning Method',
                   loc='upper center', bbox_to_anchor=(0.65, 1.01), ncol=3, fontsize=10, title_fontsize=11)

        # Layout adjustment (no clipping)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        path_dir = Path(output_dir)
        path_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(path_dir / f'time_plot.{self.figure_file_type}')
        if show:
            plt.show()
        plt.close(fig)

    def get_ensemble_weights(
        self,
        df_results: pd.DataFrame,
        method: str,
        excluded_families: list[str] = None,
        aggregate_folds: bool = False,
    ) -> pd.DataFrame:
        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if excluded_families is None:
            excluded_families = []

        df_results_method = df_results[df_results[self.method_col] == method]

        df_ensemble_weights = df_results_method[["dataset", "fold", "ensemble_weight"]]

        full_dict = []
        # available_configs = set()
        # for ensemble_weights in df_ensemble_weights["ensemble_weight"].values:
        #     for k in ensemble_weights.keys():
        #         if k not in available_configs:
        #             available_configs.add(k)
        #     ens_weights_w_dataset_fold = ensemble_weights.copy(deep=True)
        #     full_dict.append(ensemble_weights)
        #
        # df_ensemble_weights_2 = pd.DataFrame()

        for d, f, ensemble_weights in zip(df_ensemble_weights["dataset"], df_ensemble_weights["fold"], df_ensemble_weights["ensemble_weight"]):
            if isinstance(ensemble_weights, str):
                ensemble_weights = json.loads(ensemble_weights)
            assert isinstance(ensemble_weights, dict)
            ens_weights_w_dataset_fold = dict()
            ens_weights_w_dataset_fold["dataset"] = d
            ens_weights_w_dataset_fold["fold"] = f
            ens_weights_w_dataset_fold.update(ensemble_weights)
            full_dict.append(ens_weights_w_dataset_fold)
            pass

        model_to_families = self.config_types

        model_families = set()
        for m, f in model_to_families.items():
            if f not in model_families:
                model_families.add(f)

        weight_per_family_dict = []
        for cur_dict in full_dict:
            new_dict = {}
            for k, v in cur_dict.items():
                if k == "dataset":
                    new_dict["dataset"] = v
                elif k == "fold":
                    new_dict["fold"] = v
                else:
                    model_family = model_to_families[k]
                    if model_family not in new_dict:
                        new_dict[model_family] = 0
                    new_dict[model_family] += v
            weight_per_family_dict.append(new_dict)

        import pandas as pd
        df = pd.DataFrame(weight_per_family_dict)
        df = df.set_index(["dataset", "fold"])
        df = df.fillna(0)

        df_cols = df.columns
        f_to_add = []
        for f in model_families:
            if f not in df_cols:
                f_to_add.append(f)
        df[f_to_add] = 0

        if excluded_families:
            df = df.drop(columns=excluded_families)

        df = self._get_ensemble_weights(
            df_ensemble_weights=df,
            aggregate_folds=aggregate_folds,
            sort_by_mean=True,
        )

        return df

    @classmethod
    def _get_ensemble_weights(
        cls,
        df_ensemble_weights: pd.DataFrame,
        aggregate_folds: bool = True,
        sort_by_mean: bool = True,
    ) -> pd.DataFrame:
        df_ensemble_weights = copy.deepcopy(df_ensemble_weights)
        if aggregate_folds:
            df_ensemble_weights = df_ensemble_weights.groupby(level='dataset').mean()
        else:
            index_new = list(df_ensemble_weights.index.to_flat_index())
            index_new = [str(t[0]) + "_" + str(t[1]) for t in index_new]
            df_ensemble_weights.index = index_new

        if sort_by_mean:
            s = df_ensemble_weights.sum()
            df_ensemble_weights = df_ensemble_weights[s.sort_values(ascending=False).index]
        return df_ensemble_weights


    # TODO: aggregate_config_family: bool
    # TODO: sort rows by size? color by problem type?
    def _plot_ensemble_weights_heatmap(
        self,
        df_ensemble_weights: pd.DataFrame,
        aggregate_folds: bool = True,
        sort_by_mean: bool = True,
        include_mean: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        df_ensemble_weights : pd.DataFrame
            The 2nd output object of `repo.evaluate_ensembles(...)
        aggregate_folds : bool, default True
            If True, averages folds of datasets together into single rows representing a dataset.
            If False, each fold of each dataset will be its own row.
        sort_by_mean : bool, default True
            If True, will sort columns by the mean value of the column.
            If False, columns will remain in the original order.
        include_mean : bool, default True
            If True, will add a row at the bottom with label "mean" representing the mean of the config weights across all tasks.
            NaN values are considered 0 for the purposes of calculating the mean.
        **kwargs
            Passed to the `create_heatmap` function

        Returns
        -------
        plt

        """
        # df_ensemble_weights = self.get_ensemble_weights(
        #     df_ensemble_weights=df_ensemble_weights,
        #     aggregate_folds=aggregate_folds,
        #     sort_by_mean=sort_by_mean,
        # )

        p = create_heatmap(df=df_ensemble_weights, include_mean=include_mean, **kwargs)
        return p

    def plot_ensemble_weights_heatmap(self, df_ensemble_weights: pd.DataFrame, **kwargs):
        # FIXME: if family never present, then this won't work
        p = self._plot_ensemble_weights_heatmap(df_ensemble_weights=df_ensemble_weights, **kwargs)
        fig_path = Path(f"{self.output_dir}/figures")
        fig_path.mkdir(parents=True, exist_ok=True)
        p.savefig(fig_path / f"ens-weights-per-dataset.{self.figure_file_type}")

    # FIXME: clean this up
    def generate_runtime_plot(self, df_results: pd.DataFrame):
        from scripts.dataset_analysis import plot_train_time_deep_dive  # FIXME
        df_results_configs = df_results[df_results["method_type"] == "config"]
        df_results_configs = df_results_configs.copy(deep=True)

        framework_types = self._get_config_types(df_results=df_results_configs)
        df_results_configs = df_results_configs[df_results_configs["config_type"].isin(framework_types)]

        method_rename_map = self.get_method_rename_map()
        df_results_configs["config_type"] = df_results_configs["config_type"].map(method_rename_map)

        plot_train_time_deep_dive(
            df=df_results_configs,
            expname_outdir=self.output_dir,
            method_col=self.method_col,
            family_col="config_type",
            show=False
        )
