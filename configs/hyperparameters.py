from src.pso import PSOVectorized, PSOLVIW, APSO, PSOTVAC, APSOVI, UAPSO

N_PARTICLES = 50

CONFIG_HYPERPARAMETERS = {
    "PSO": {"w": 0.8, "c1": 2.0, "c2": 2.0, "n_particles": N_PARTICLES},
    "PSOLVIW": {
        "w_max": 0.9,
        "w_min": 0.4,
        "c1": 2.0,
        "c2": 2.0,
        "n_particles": N_PARTICLES,
    },
    "PSOTVAC": {
        "w_max": 0.9,
        "w_min": 0.4,
        "c1_max": 2.5,
        "c1_min": 0.5,
        "c2_max": 2.5,
        "c2_min": 0.5,
        "n_particles": N_PARTICLES,
    },
    "APSOVI": {
        "w_max": 0.9,
        "w_min": 0.3,
        "step_size": 0.1,
        "c1": 1.496180,
        "c2": 1.496180,
        "n_particles": N_PARTICLES,
    },
    "APSO": {"w": 0.9, "c1": 2.0, "c2": 2.0, "n_particles": N_PARTICLES},
    "UAPSO": {
        "w_min": 0.0,
        "w_max": 1.0,
        "c1_min": 0.0,
        "c1_max": 2.0,
        "c2_min": 0.0,
        "c2_max": 2.0,
        "learning_rate": 0.01,
        "threshold": 0.5,
        "n_particles": N_PARTICLES,
    },
}


ALGORITHMS = {
    "PSO": PSOVectorized,
    "PSOLVIW": PSOLVIW,
    "PSOTVAC": PSOTVAC,
    "APSOVI": APSOVI,
    "APSO": APSO,
    "UAPSO": UAPSO,
}
