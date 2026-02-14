import pandas as pd
from experiments import CONFIG_EXPERIMENTS


def main():
    df_results = pd.DataFrame()

    for dim in CONFIG_EXPERIMENTS["dim"]:
        df = pd.read_pickle(f"results/benchmarks_{dim}dim_results.pkl")

        df_results = pd.concat([df_results, df], axis=0)

    df_results.to_csv("results/benchmarks_results.csv", index=False)
    df_results.to_pickle("results/benchmarks_results.pkl")


if __name__ == "__main__":
    main()
