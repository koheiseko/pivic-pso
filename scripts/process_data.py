import pandas as pd
from pathlib import Path
from scripts.run_experiments import CONFIG_EXPERIMENTS


def main(results_dir: str = "results"):
    base_path = Path(results_dir)
    dataframes = []

    dimensions = CONFIG_EXPERIMENTS.get("dim", [])

    for dim in dimensions:
        file_path = base_path / f"benchmarks_{dim}dim_results.pkl"

        if file_path.exists():
            df = pd.read_pickle(file_path)
            dataframes.append(df)
        else:
            print(f"[AVISO] Arquivo não encontrado: {file_path.name}.")

    if not dataframes:
        print(
            "\nErro: Nenhum dado foi encontrado para processar. Verifique a pasta de resultados."
        )
        return

    df_results = pd.concat(dataframes, ignore_index=True)

    path_csv_out = base_path / "benchmarks_results.csv"
    path_pkl_out = base_path / "benchmarks_results.pkl"

    df_results.to_csv(path_csv_out, index=False)
    df_results.to_pickle(path_pkl_out)


if __name__ == "__main__":
    main()
