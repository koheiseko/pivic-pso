import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Plotter:
    def __init__(self):
        self.colors = sns.color_palette("bright", 10)
        self.line_styles = ["-", "--", "-.", ":"]

    def plot_convergence(
        self, df: pd.DataFrame, log_scale: bool = True, output_path: str = None
    ):
        benchmarks = df["benchmark"].unique()
        dimensions = df["dim"].unique()
        algorithms = df["algorithm"].unique()

        n_benchmarks = len(benchmarks)
        n_dimensions = len(dimensions)

        fig, axes = plt.subplots(
            n_benchmarks, n_dimensions, figsize=(12, 12), sharex=True
        )

        if n_benchmarks == 1 or n_dimensions == 1:
            import numpy as np

            axes = np.atleast_2d(axes)

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
                        mean_fitness = df_subplot["mean_fitness_history"].item()

                        if log_scale:
                            (line,) = ax.loglog(
                                mean_fitness,
                                linewidth=1.5,
                                linestyle=self.line_styles[a],
                                color=self.colors[a],
                            )

                        (line,) = ax.plot(
                            mean_fitness,
                            linewidth=1.5,
                            linestyle=self.line_styles[a],
                            color=self.colors[a],
                        )

                        if i == 0 and j == 0:
                            lines.append(line)

                ax.set_title(f"{benchmark.capitalize()} (dim = {dim})", fontsize=12)

        fig.supxlabel("Iterations")
        fig.supylabel("Fitness")

        fig.legend(
            lines,
            algorithms,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(algorithms),
            frameon=False,
            fontsize="medium",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(output_path, bbox_inches="tight", transparent=False)

    def plot_parameter_dynamics(
        self, df, param_name="w", log_scale=False, output_path: str = None
    ):
        benchmarks = df["benchmark"].unique()
        dimensions = df["dim"].unique()
        algorithms = df["algorithm"].unique()

        n_benchmarks = len(benchmarks)
        n_dimensions = len(dimensions)

        fig, axes = plt.subplots(
            n_benchmarks, n_dimensions, figsize=(12, 12), sharex=True
        )

        if n_benchmarks == 1 or n_dimensions == 1:
            import numpy as np

            axes = np.atleast_2d(axes)

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
                        mean_fitness = df_subplot[
                            "mean_" + param_name + "_history"
                        ].item()

                        if log_scale:
                            (line,) = ax.loglog(
                                mean_fitness,
                                linewidth=1.5,
                                linestyle=self.line_styles[a],
                                color=self.colors[a],
                            )

                        (line,) = ax.plot(
                            mean_fitness,
                            linewidth=1.5,
                            linestyle=self.line_styles[a],
                            color=self.colors[a],
                        )

                        if i == 0 and j == 0:
                            lines.append(line)

                ax.set_title(f"{benchmark.capitalize()} (dim = {dim})", fontsize=12)

        fig.supxlabel("Iterations")
        fig.supylabel(param_name)

        fig.legend(
            lines,
            algorithms,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(algorithms),
            frameon=False,
            fontsize="medium",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(output_path, bbox_inches="tight", transparent=False)


if __name__ == "__main__":
    df = pd.read_pickle("results/benchmarks_results.pkl")

    plotter = Plotter()

    output_path = "results/convergence.pdf"
    plotter.plot_convergence(df, log_scale=False, output_path=output_path)

    output_path = "results/convergence_w.pdf"
    plotter.plot_parameter_dynamics(
        df, param_name="w", log_scale=False, output_path=output_path
    )
    output_path = "results/convergence_c1.pdf"
    plotter.plot_parameter_dynamics(
        df, log_scale=False, param_name="c1", output_path=output_path
    )
    output_path = "results/convergence_c2.pdf"
    plotter.plot_parameter_dynamics(
        df, log_scale=False, param_name="c2", output_path=output_path
    )
