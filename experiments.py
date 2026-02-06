import numpy as np
from configs.benchmarks import CONFIG_BENCHMARKS
from configs.hyperparameters import CONFIG_HYPERPARAMETERS, ALGORITHMS
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

np.random.seed(42)


config_experiments = {
    "n_runs": 20,
    "n_iterations": 100,
    "dim": [30, 50, 100, 500],
    "algorithms": ALGORITHMS,
    "config_hyperparameters": CONFIG_HYPERPARAMETERS,
    "config_benchmarks": CONFIG_BENCHMARKS,
}


def run_experiments(
    configs: Dict, output_file: str = "results/benchmarks_results.pkl"
) -> List[Dict]:
    results = []
    n_runs = configs["n_runs"]
    n_iterations = configs["n_iterations"]

    for dim in configs["dim"]:
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
                    f"--- Iniciando a execução: {algorithm_name}, {benchmark_name.capitalize()}, {dim} DIM, {configs['n_iterations']} Iterations, {configs['n_runs']} Runs ---"
                )

                all_fitness = []
                all_best_fitness = []
                all_w = []
                all_c1 = []
                all_c2 = []

                for _ in tqdm(range(n_runs), desc=f"Executando {algorithm_name}"):
                    optimizer = algorithm(**hyperparameters)
                    best_fitness, _, history = optimizer.optimize(
                        function, n_iterations
                    )

                    all_best_fitness.append(best_fitness)
                    all_fitness.append(history["fitness"])
                    all_w.append(history["w"])
                    all_c1.append(history["c1"])
                    all_c2.append(history["c2"])

                results.extend(
                    [
                        {
                            "benchmark": benchmark_name,
                            "dim": dim,
                            "algorithm": algorithm_name,
                            "best_fitness_history": np.array(all_best_fitness),
                            "mean_best_fitness": np.array(all_best_fitness).mean(),
                            "std_best_fitness": np.array(all_best_fitness).std(),
                            "mean_fitness_history": np.array(all_fitness).mean(axis=0),
                            "std_fitness_history": np.array(all_fitness).std(axis=0),
                            "mean_w_history": np.array(all_w).mean(axis=0),
                            "mean_c1_history": np.array(all_c1).mean(axis=0),
                            "mean_c2_history": np.array(all_c2).mean(axis=0),
                        }
                    ]
                )

    df_results = pd.DataFrame(results)
    df_results.to_pickle(output_file)

    print(f"Dados salvos em '{output_file}'.")

    return results


if __name__ == "__main__":
    output_file = "results/benchmarks_results.pkl"

    run_experiments(config_experiments, output_file=output_file)
