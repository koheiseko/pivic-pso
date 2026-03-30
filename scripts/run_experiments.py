import numpy as np
import pandas as pd
from typing import Dict
from joblib import Parallel, delayed
import os

from configs.benchmarks import CONFIG_BENCHMARKS
from configs.hyperparameters import CONFIG_HYPERPARAMETERS, ALGORITHMS

BASE_SEED = 42

CONFIG_EXPERIMENTS = {
    "n_runs": 50,
    "n_iterations": 5000,
    "dim": [30, 50, 100],
    "algorithms": ALGORITHMS,
    "config_hyperparameters": CONFIG_HYPERPARAMETERS,
    "config_benchmarks": CONFIG_BENCHMARKS,
    "n_jobs": -1,
}


def single_run(run_id: int, algorithm, hyperparameters, function, n_iterations) -> Dict:
    current_seed = BASE_SEED + run_id
    np.random.seed(current_seed)

    try:
        optimizer = algorithm(**hyperparameters)

        best_fitness, _, history = optimizer.optimize(function, n_iterations)

        return {
            "success": True,
            "best_fitness": best_fitness,
            "fitness_history": history["fitness"],
            "w_history": history.get("w", []),
            "c1_history": history.get("c1", []),
            "c2_history": history.get("c2", []),
        }

    except Exception as e:
        print(f"[WARN] Erro na Run {run_id}: {e}")
        return {"success": False}


def run_experiments(configs: Dict, output_dir: str = "results") -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_runs = configs["n_runs"]
    n_iterations = configs["n_iterations"]
    n_jobs = configs.get("n_jobs", 1)

    for dim in configs["dim"]:
        dim_results = []

        for benchmark_name, benchmark in configs["config_benchmarks"].items():
            for algorithm_name, algorithm in configs["algorithms"].items():
                bounds = benchmark["bounds"]
                function = benchmark["function"]

                hyperparameters = configs["config_hyperparameters"][
                    algorithm_name
                ].copy()
                hyperparameters["dim"] = dim
                hyperparameters["low"] = bounds[0]
                hyperparameters["high"] = bounds[1]

                print(
                    f"[INFO] Iniciando a execução: {algorithm_name}, {benchmark_name.capitalize()}, {dim} DIM, {configs['n_iterations']} Iterations, {configs['n_runs']} Runs"
                )

                runs_output = Parallel(n_jobs=n_jobs)(
                    delayed(single_run)(
                        i, algorithm, hyperparameters, function, n_iterations
                    )
                    for i in range(n_runs)
                )

                successful_runs = [r for r in runs_output if r["success"]]

                if not successful_runs:
                    print(
                        f"[WARN] Todas as execuções falharam para {algorithm_name} em {benchmark_name}"
                    )
                    continue

                all_best_fitness = [r["best_fitness"] for r in successful_runs]
                all_fitness = [r["fitness_history"] for r in successful_runs]

                all_w = [
                    r["w_history"] for r in successful_runs if len(r["w_history"]) > 0
                ]
                all_c1 = [
                    r["c1_history"] for r in successful_runs if len(r["c1_history"]) > 0
                ]
                all_c2 = [
                    r["c2_history"] for r in successful_runs if len(r["c2_history"]) > 0
                ]

                result_entry = {
                    "benchmark": benchmark_name,
                    "dim": dim,
                    "algorithm": algorithm_name,
                    "best_fitness_history": np.array(all_best_fitness),
                    "mean_best_fitness": np.mean(all_best_fitness),
                    "std_best_fitness": np.std(all_best_fitness),
                    "mean_fitness_history": np.mean(all_fitness, axis=0),
                    "std_fitness_history": np.std(all_fitness, axis=0),
                }

                if all_w:
                    result_entry["mean_w_history"] = np.mean(all_w, axis=0)
                if all_c1:
                    result_entry["mean_c1_history"] = np.mean(all_c1, axis=0)
                if all_c2:
                    result_entry["mean_c2_history"] = np.mean(all_c2, axis=0)

                dim_results.append(result_entry)

        df_results = pd.DataFrame(dim_results)
        filename = f"{output_dir}/benchmarks_{dim}dim_results.pkl"

        df_summary = df_results.drop(
            columns=[
                "mean_fitness_history",
                "std_fitness_history",
                "mean_w_history",
                "mean_c1_history",
                "mean_c2_history",
            ],
            errors="ignore",
        )

        try:
            df_results.to_pickle(filename)
            df_summary.to_csv(filename.replace(".pkl", "_summary.csv"), index=False)
            print(f"[INFO] Salvo: {filename}")
        except Exception as e:
            print(f"[WARN] Erro ao salvar arquivo: {e}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_experiments(CONFIG_EXPERIMENTS)

    print(time.time() - start_time)
