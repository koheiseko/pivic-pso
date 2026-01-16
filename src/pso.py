import numpy as np
from typing import List, Tuple, Callable


class Particle:
    def __init__(self, low: float, high: float, dim: int):
        self.dim = dim
        self.position = np.random.uniform(low, high, dim)
        self.velocity = np.zeros(dim)
        self.pbest_position = self.position.copy()
        self.pbest_value = float("inf")
        self.value = 0

    def fitness(self, function: Callable) -> None:
        self.value = function(self.position)

        if self.value < self.pbest_value:
            self.pbest_value = self.value
            self.pbest_position = self.position.copy()

    def update_velocity(
        self, gbest_position: np.ndarray, w: float, c1: float, c2: float
    ) -> None:
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
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w: float,
        c1: float,
        c2: float,
        dim: int = 2,
    ):
        self.n_particles = n_particles

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = [Particle(low, high, dim) for _ in range(n_particles)]

        self.gbest_value = float("inf")
        self.gbest_position = np.zeros(dim)

    def optimize(
        self, function: Callable, n_epochs: int
    ) -> Tuple[float, np.ndarray, List[np.ndarray]]:
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

        return (self.gbest_value, self.gbest_position, history)

    def __repr__(self):
        return f"PSO(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class PSOVectorized:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w: float,
        c1: float,
        c2: float,
        dim: int,
    ):
        self.n_particles = n_particles

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.dim = dim
        self.low = low
        self.high = high

        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

    def optimize(
        self, function: Callable, n_epochs: int
    ) -> Tuple[float, np.ndarray, List[float]]:
        history = []

        for epoch in range(n_epochs):
            current_values = np.apply_along_axis(
                function, axis=1, arr=self.particles_positions
            )

            mask_better = current_values < self.pbest_values

            self.pbest_values[mask_better] = current_values[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(current_values)

            if current_values[min_idx] < self.gbest_value:
                self.gbest_value = current_values[min_idx]
                self.gbest_position = self.particles_positions[min_idx].copy()

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = self.w * self.particles_velocities
            cognitive = self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = self.c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            history.append(self.gbest_value)

        return self.gbest_value, self.gbest_position, history

    def __repr__(self):
        return f"PSOVectorized(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class PSOLVIW:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_max: float,
        w_min: float,
        c1: float,
        c2: float,
        dim: int,
    ):
        self.n_particles = n_particles

        self.w_max = w_max
        self.w_min = w_min

        self.w = w_max
        self.c1 = c1
        self.c2 = c2

        self.dim = dim
        self.low = low
        self.high = high

        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

    def optimize(
        self, function: Callable, n_epochs: int
    ) -> Tuple[float, np.ndarray, List[np.ndarray]]:
        history = []

        for epoch in range(n_epochs):
            current_values = np.apply_along_axis(
                function, axis=1, arr=self.particles_positions
            )

            mask_better = current_values < self.pbest_values

            self.pbest_values[mask_better] = current_values[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(current_values)

            if current_values[min_idx] < self.gbest_value:
                self.gbest_value = current_values[min_idx]
                self.gbest_position = self.particles_positions[min_idx].copy()

            self.w = self.w_max - ((self.w_max - self.w_min) * epoch) / n_epochs

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = self.w * self.particles_velocities
            cognitive = self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = self.c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            history.append(self.gbest_value)

        return self.gbest_value, self.gbest_position, history

    def __repr__(self):
        return f"PSOVectorized(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"
