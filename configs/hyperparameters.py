from src.pso import PSOVectorized, PSOLVIW, APSO

W = 0.9
W_MIN = 0.4
C1 = 2.0
C2 = 2.0
DIM = 2
N_PARTICLES = 100

CONFIG_HYPERPARAMETERS = {
    "PSO": {"w": W, "c1": C1, "c2": C2, "n_particles": N_PARTICLES},
    "PSOLVIW": {
        "w_max": W,
        "w_min": W_MIN,
        "c1": C1,
        "c2": C2,
        "n_particles": N_PARTICLES,
    },
    "APSO": {"w": W, "c1": C1, "c2": C2, "n_particles": N_PARTICLES},
}

ALGORITHMS = {"PSO": PSOVectorized, "PSOLVIW": PSOLVIW, "APSO": APSO}
