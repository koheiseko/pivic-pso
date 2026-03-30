import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pathlib import Path


class Plotter:
    def __init__(self):
        self.colors = sns.color_palette("bright", 10)
        self.line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

        self.alg_names = {
            "PSOLVIW": "PSO-LVIW",
            "PSOTVAC": "PSO-TVAC",
            "APSOVI": "APSO-VI",
            "APSO": "APSO",
            "UAPSO": "UAPSO-A",
            "PSO": "PSO",
        }

        self.func_map = {
            "sphere": "f1",
            "rastrigin": "f2",
            "ackley": "f3",
            "rosenbrock": "f4",
        }

    def _plot_line_with_std(
        self,
        ax,
        iterations,
        mean_vals,
        std_vals,
        alg_index,
        alg_name,
        log_scale,
        alpha: float = 1,
    ):
        color = self.colors[alg_index % len(self.colors)]
        style = self.line_styles[alg_index % len(self.line_styles)]
        name_formated = self.alg_names.get(alg_name, alg_name)

        if log_scale:
            (line,) = ax.loglog(
                iterations,
                mean_vals,
                linewidth=1.5,
                linestyle=style,
                color=color,
                label=name_formated,
                alpha=alpha,
            )
        else:
            (line,) = ax.plot(
                iterations,
                mean_vals,
                linewidth=1.5,
                linestyle=style,
                color=color,
                label=name_formated,
                alpha=alpha,
            )

        if std_vals is not None:
            ax.fill_between(
                iterations,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.15,
            )
        return line

    def plot_single_convergence(
        self,
        df: pd.DataFrame,
        benchmark: str,
        dim: int,
        output_path: Path,
        log_scale: bool = False,
    ):
        df_filtered = df[(df["benchmark"] == benchmark) & (df["dim"] == dim)]
        if df_filtered.empty:
            print(f"Aviso: Sem dados para {benchmark} ({dim}D).")
            return

        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        algorithms = df_filtered["algorithm"].unique()

        for a, algorithm in enumerate(algorithms):
            df_alg = df_filtered[df_filtered["algorithm"] == algorithm]
            mean_vals = np.array(df_alg["mean_fitness_history"].item())
            std_vals = np.array(df_alg["std_fitness_history"].item())
            iterations = np.arange(len(mean_vals))

            self._plot_line_with_std(
                ax, iterations, mean_vals, std_vals, a, algorithm, log_scale
            )

        sns.despine()
        plt.legend(loc="upper right", frameon=True)
        plt.ylabel("Fitness")
        plt.xlabel("Iterations")

        title_func = self.func_map.get(benchmark, benchmark)
        plt.title(f"{title_func}, Dim: {dim}")
        plt.tight_layout()

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def plot_all_convergence(
        self, df: pd.DataFrame, log_scale: bool = False, output_path: Path = None
    ):
        benchmarks = df["benchmark"].unique()
        dimensions = df["dim"].unique()
        algorithms = df["algorithm"].unique()

        fig, axes = plt.subplots(
            len(benchmarks),
            len(dimensions),
            figsize=(12, 12),
            sharex=True,
            squeeze=False,
        )
        lines = []

        for i, benchmark in enumerate(benchmarks):
            for j, dim in enumerate(dimensions):
                ax = axes[i, j]
                for a, algorithm in enumerate(algorithms):
                    df_subplot = df[
                        (df["benchmark"] == benchmark)
                        & (df["dim"] == dim)
                        & (df["algorithm"] == algorithm)
                    ]

                    if not df_subplot.empty:
                        mean_fitness = np.array(
                            df_subplot["mean_fitness_history"].item()
                        )
                        std_vals = np.array(df_subplot["std_fitness_history"].item())
                        iterations = np.arange(len(mean_fitness))

                        line = self._plot_line_with_std(
                            ax,
                            iterations,
                            mean_fitness,
                            std_vals,
                            a,
                            algorithm,
                            log_scale,
                        )

                        if i == 0 and j == 0:
                            lines.append(line)

                title_func = self.func_map.get(benchmark, benchmark)
                ax.set_title(f"{title_func}, Dim: {dim}", fontsize=12)

        fig.supxlabel("Iterations")
        fig.supylabel("Fitness")
        format_algs = [self.alg_names.get(alg, alg) for alg in algorithms]

        fig.legend(
            lines,
            format_algs,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(algorithms),
            frameon=False,
            fontsize="medium",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", transparent=False)
            plt.close()

    def plot_parameter_dynamics(
        self,
        df: pd.DataFrame,
        param_name="w",
        log_scale=False,
        output_path: Path = None,
    ):
        benchmarks = df["benchmark"].unique()
        dimensions = df["dim"].unique()
        algorithms = df["algorithm"].unique()

        fig, axes = plt.subplots(
            len(benchmarks),
            len(dimensions),
            figsize=(12, 12),
            sharex=True,
            squeeze=False,
        )
        lines = []

        for i, benchmark in enumerate(benchmarks):
            for j, dim in enumerate(dimensions):
                ax = axes[i, j]
                for a, algorithm in enumerate(algorithms):
                    df_subplot = df[
                        (df["benchmark"] == benchmark)
                        & (df["dim"] == dim)
                        & (df["algorithm"] == algorithm)
                    ]

                    if not df_subplot.empty:
                        mean_param = np.array(
                            df_subplot["mean_" + param_name + "_history"].item()
                        )
                        iterations = np.arange(len(mean_param))

                        std_col = "std_" + param_name + "_history"
                        if std_col in df_subplot.columns:
                            std_param = np.array(df_subplot[std_col].item())
                        else:
                            std_param = None

                        line = self._plot_line_with_std(
                            ax,
                            iterations,
                            mean_param,
                            std_param,
                            a,
                            algorithm,
                            log_scale,
                            alpha=0.5,
                        )

                        if i == 0 and j == 0:
                            lines.append(line)

                title_func = self.func_map.get(benchmark, benchmark)
                ax.set_title(f"{title_func}, Dim: {dim}", fontsize=12)

        fig.supxlabel("Iterations")

        ylabel = param_name
        if param_name == "c1":
            ylabel = "$c_1$"
        elif param_name == "c2":
            ylabel = "$c_2$"
        elif param_name == "w":
            ylabel = "$w$"
        fig.supylabel(ylabel)

        format_algs = [self.alg_names.get(alg, alg) for alg in algorithms]
        fig.legend(
            lines,
            format_algs,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(algorithms),
            frameon=False,
            fontsize="medium",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", transparent=False)
            plt.close()
