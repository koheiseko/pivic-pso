import numpy as np


def rastrigin(x: np.ndarray) -> float:
    dim = len(x)
    return 10 * dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def ackley(x: np.ndarray) -> float:
    dim = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / dim)

    return term1 + term2 + 20 + np.e
