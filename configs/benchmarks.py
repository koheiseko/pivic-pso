from src.functions import sphere, ackley, rosenbrock, rastrigin

CONFIG_BENCHMARKS = {
    "sphere": {"bounds": (-100, 100), "function": sphere},
    "rastrigin": {"bounds": (-5.12, 5.12), "function": rastrigin},
    "ackley": {"bounds": (-32.00, 32.00), "function": ackley},
    "rosenbrock": {"bounds": (-10.0, 10.0), "function": rosenbrock},
}
