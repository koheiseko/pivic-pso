import pandas as pd

from pathlib import Path

from src.plotter import Plotter


def main():
    input_filepath = Path("../results/benchmarks_results.pkl")

    df = pd.read_pickle(input_filepath)
    plotter = Plotter()

    base_figs_dir = Path("../reports/figures")
    single_dir = base_figs_dir / "single_convergence"
    all_dir = base_figs_dir / "all_convergence"
    param_dir = base_figs_dir / "parameter_dynamics"

    single_dir.mkdir(parents=True, exist_ok=True)
    all_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)

    single_plots = [
        ("rastrigin", 30, False, "convergence_rastrigin_30.pdf"),
        ("rastrigin", 50, False, "convergence_rastrigin_50.pdf"),
        ("rastrigin", 100, False, "convergence_rastrigin_100.pdf"),
        ("ackley", 30, False, "convergence_ackley_30.pdf"),
        ("ackley", 50, False, "convergence_ackley_50.pdf"),
        ("ackley", 100, False, "convergence_ackley_100.pdf"),
        ("rosenbrock", 30, False, "convergence_rosenbrock_30.pdf"),
        ("rosenbrock", 50, False, "convergence_rosenbrock_50.pdf"),
        ("rosenbrock", 100, False, "convergence_rosenbrock_100.pdf"),
        ("sphere", 30, False, "convergence_sphere_30.pdf"),
        ("sphere", 50, False, "convergence_sphere_50.pdf"),
        ("sphere", 100, False, "convergence_sphere_100.pdf"),
    ]

    for benchmark, dim, log_scale, filename in single_plots:
        out_path = single_dir / filename
        plotter.plot_single_convergence(
            df, benchmark=benchmark, dim=dim, output_path=out_path, log_scale=log_scale
        )

    grid_conv_path = all_dir / "grid_convergence_all.pdf"
    plotter.plot_all_convergence(df, log_scale=False, output_path=grid_conv_path)

    for param in ["w", "c1", "c2"]:
        filename = f"grid_dynamics_{param}.pdf"
        out_path = param_dir / filename

        plotter.plot_parameter_dynamics(
            df, param_name=param, log_scale=False, output_path=out_path
        )


if __name__ == "__main__":
    main()
