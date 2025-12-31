import numpy as np


class Particle:
    def __init__(self, low, high, dim):
        self.dim = dim
        self.position = np.random.uniform(low, high, self.dim)
        self.velocity = np.zeros(self.dim)
        self.pbest_position = self.position
        self.pbest_value = float("inf")
        self.value = 0

    def fitness(self, function) -> None:
        self.value = function(self.position)

        if self.value < self.pbest_value:
            self.pbest_value = self.value
            self.pbest_position = self.position.copy()

    def update_velocity(self, gbest_position, w, c1, c2) -> None:
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.pbest_position - self.position)
            + c2 * r2 * (gbest_position - self.position)
        )

    def update_position(self) -> None:
        self.position = self.position + self.velocity

    def __repr__(self):
        return f"Particle(position={self.position}, velocity={self.velocity})"


class PSO:
    def __init__(self, n_particles, low, high, w, c1, c2, dim=2):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = [Particle(low, high, dim) for _ in range(n_particles)]
        self.gbest_value = float("inf")
        self.gbest_position = None

    def optimize(self, function, n_epochs):
        history = []
        for epoch in range(n_epochs):
            for particle in self.particles:
                particle.fitness(function)

                if particle.value < self.gbest_value:
                    self.gbest_value = particle.value
                    self.gbest_position = particle.position.copy()

            for particle in self.particles:
                particle.update_velocity(
                    self.gbest_position, w=self.w, c1=self.c1, c2=self.c2
                )
                particle.update_position()

            snapshot = np.array([p.position.copy() for p in self.particles])
            history.append(snapshot)

        return self.gbest_value, self.gbest_position, history