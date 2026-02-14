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
        self, function: Callable, n_iterations: int
    ) -> Tuple[float, np.ndarray, List[np.ndarray]]:
        history = []

        for iteration in range(n_iterations):
            for particle in self.particles:
                particle.fitness(function)

                if particle.pbest_value < self.gbest_value:
                    self.gbest_value = particle.pbest_value
                    self.gbest_position = particle.pbest_position.copy()

            for particle in self.particles:
                particle.update_velocity(
                    self.gbest_position, w=self.w, c1=self.c1, c2=self.c2
                )
                particle.update_position()

            history.append(self.gbest_value)

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

        self.v_max = 0.2 * (high - low)

        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self, function: Callable, n_iterations: int
    ) -> Tuple[float, np.ndarray, List[float]]:
        self.history = {"fitness": [], "w": [], "c1": [], "c2": []}

        for iteration in range(n_iterations):
            self.current_fitness = np.apply_along_axis(
                function, axis=1, arr=self.particles_positions
            )

            mask_better = self.current_fitness < self.pbest_values

            self.pbest_values[mask_better] = self.current_fitness[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(self.pbest_values)

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = self.pbest_values[min_idx]
                self.gbest_position = self.pbest_positions[min_idx].copy()

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = self.w * self.particles_velocities
            cognitive = self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = self.c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            self.history["fitness"].append(self.gbest_value)
            self.history["w"].append(self.w)
            self.history["c1"].append(self.c1)
            self.history["c2"].append(self.c2)

        return self.gbest_value, self.gbest_position, self.history

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

        self.v_max = 0.2 * (high - low)

        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self, function: Callable, n_iterations: int
    ) -> Tuple[float, np.ndarray, List[np.ndarray]]:
        self.history = {"fitness": [], "w": [], "c1": [], "c2": []}

        for iteration in range(n_iterations):
            self.current_fitness = np.apply_along_axis(
                function, axis=1, arr=self.particles_positions
            )

            mask_better = self.current_fitness < self.pbest_values

            self.pbest_values[mask_better] = self.current_fitness[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(self.pbest_values)

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = self.pbest_values[min_idx]
                self.gbest_position = self.pbest_positions[min_idx].copy()

            self.w = (
                self.w_max
                - ((self.w_max - self.w_min) * (iteration + 1)) / n_iterations
            )

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = self.w * self.particles_velocities
            cognitive = self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = self.c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            self.history["fitness"].append(self.gbest_value)
            self.history["w"].append(self.w)
            self.history["c1"].append(self.c1)
            self.history["c2"].append(self.c2)

        return self.gbest_value, self.gbest_position, self.history

    def __repr__(self):
        return f"PSOLVIW(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class PSOTVAC:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_min: float,
        w_max: float,
        c1_min: float,
        c1_max: float,
        c2_min: float,
        c2_max: float,
        dim: int,
    ):
        self.n_particles = n_particles

        self.w_min = w_min
        self.w_max = w_max

        self.c1_min = c1_min
        self.c1_max = c1_max

        self.c2_min = c2_min
        self.c2_max = c2_max

        self.dim = dim
        self.low = low
        self.high = high

        self.v_max = 0.2 * (high - low)

        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self, function: Callable, n_iterations: int
    ) -> Tuple[float, np.ndarray, List[float]]:
        self.history = {"fitness": [], "w": [], "c1": [], "c2": []}

        for iteration in range(n_iterations):
            self.c1 = (self.c1_min - self.c1_max) * (
                iteration / n_iterations
            ) + self.c1_max
            self.c2 = (self.c2_max - self.c2_min) * (
                iteration / n_iterations
            ) + self.c2_min
            self.w = (self.w_max - self.w_min) * (
                (n_iterations - iteration) / n_iterations
            ) + self.w_min

            self.current_fitness = np.apply_along_axis(
                function, axis=1, arr=self.particles_positions
            )

            mask_better = self.current_fitness < self.pbest_values

            self.pbest_values[mask_better] = self.current_fitness[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(self.pbest_values)

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = self.pbest_values[min_idx]
                self.gbest_position = self.pbest_positions[min_idx].copy()

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = self.w * self.particles_velocities
            cognitive = self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = self.c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            self.history["fitness"].append(self.gbest_value)
            self.history["w"].append(self.w)
            self.history["c1"].append(self.c1)
            self.history["c2"].append(self.c2)

        return self.gbest_value, self.gbest_position, self.history

    def __repr__(self):
        return f"PSOTVAC(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class APSOVI:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_min: float,
        w_max: float,
        step_size,
        c1: float,
        c2: float,
        dim: int,
    ):
        self.n_particles = n_particles

        self.w_max = w_max
        self.w_min = w_min
        self.w = w_max
        self.step_size = step_size
        self.c1 = c1
        self.c2 = c2

        self.dim = dim
        self.low = low
        self.high = high

        self.v_max = 0.2 * (high - low)
        self.v_inicial = (high - low) / 2.0

        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self, function: Callable, n_iterations: int
    ) -> Tuple[float, np.ndarray, List[float]]:
        self.history = {"fitness": [], "w": [], "c1": [], "c2": []}

        t_end = 0.95 * n_iterations

        for iteration in range(n_iterations):
            self.current_fitness = np.apply_along_axis(
                function, axis=1, arr=self.particles_positions
            )

            mask_better = self.current_fitness < self.pbest_values

            self.pbest_values[mask_better] = self.current_fitness[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(self.pbest_values)

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = self.pbest_values[min_idx]
                self.gbest_position = self.pbest_positions[min_idx].copy()

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = self.w * self.particles_velocities
            cognitive = self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = self.c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            v_ave = np.sum(np.absolute(self.particles_velocities)) / (
                self.n_particles * self.dim
            )

            v_ideal_next = (
                self.v_inicial * (1 + np.cos((iteration + 1) * np.pi / t_end)) / 2
            )

            if v_ave >= v_ideal_next:
                self.w = np.max([self.w - self.step_size, self.w_min])
            else:
                self.w = np.min([self.w + self.step_size, self.w_max])

            self.history["fitness"].append(self.gbest_value)
            self.history["w"].append(self.w)
            self.history["c1"].append(self.c1)
            self.history["c2"].append(self.c2)

        return self.gbest_value, self.gbest_position, self.history

    def __repr__(self):
        return f"APSOVI(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class APSO:
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
        self.v_max = 0.2 * (high - low)  

        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))
        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)
        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf
        self.current_fitness = np.full(n_particles, np.inf)

        self.current_state = "exploration"
        self.previous_state = None

    def _calculate_evolucionary_factor(self):
        dists = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            diff = self.particles_positions - self.particles_positions[i]
            d_sq = np.sum(diff**2, axis=1)
            d_euclidian = np.sqrt(d_sq)
            dists[i] = np.sum(d_euclidian) / (self.n_particles - 1)

        d_g_idx = np.argmin(self.current_fitness)
        d_g = dists[d_g_idx]
        d_max = np.max(dists)
        d_min = np.min(dists)

        if d_max == d_min:
            return 0.5

        f = (d_g - d_min) / (d_max - d_min)
        return np.clip(f, 0.0, 1.0)

    def _get_state(self, f):
        if 0 <= f <= 0.4:
            state_1 = 0
        elif 0.4 < f <= 0.6:
            state_1 = 5 * f - 2
        elif 0.6 < f <= 0.7:
            state_1 = 1
        elif 0.7 < f <= 0.8:
            state_1 = -10 * f + 8
        else:
            state_1 = 0

        # Exploitation 
        if 0 <= f <= 0.2:
            state_2 = 0
        elif 0.2 < f <= 0.3:
            state_2 = 10 * f - 2
        elif 0.3 < f <= 0.4:
            state_2 = 1
        elif 0.4 < f <= 0.6:
            state_2 = -5 * f + 3
        else:
            state_2 = 0

        # Convergence 
        if 0 <= f <= 0.1:
            state_3 = 1
        elif 0.1 < f <= 0.3:
            state_3 = -5 * f + 1.5
        else:
            state_3 = 0

        # Jumping Out 
        if 0 <= f <= 0.7:
            state_4 = 0
        elif 0.7 < f <= 0.9:
            state_4 = 5 * f - 3.5
        else:
            state_4 = 1

        scores = {
            "exploration": state_1,
            "exploitation": state_2,
            "convergence": state_3,
            "jumping_out": state_4,
        }

        current_best_state = max(scores, key=scores.get)

        if self.previous_state == "jumping_out" and scores["exploration"] > 0:
            return "exploration"

        if self.previous_state == "exploration" and scores["exploration"] > 0.5:
            return "exploration"

        return current_best_state

    def _adapt_parameters(self, state, f):
        new_w = 1 / (1 + 1.5 * np.exp(-2.6 * f))

        self.w = np.clip(new_w, 0.4, 0.9)

        delta = np.random.uniform(0.05, 0.1)

        if state == "exploration":
            self.c1 += delta
            self.c2 -= delta
        elif state == "exploitation":
            self.c1 += 0.5 * delta
            self.c2 -= 0.5 * delta
        elif state == "convergence":
            self.c1 += 0.5 * delta
            self.c2 += 0.5 * delta
        elif state == "jumping_out":
            self.c1 -= delta
            self.c2 += delta

        self.c1 = np.clip(self.c1, 1.5, 2.5)
        self.c2 = np.clip(self.c2, 1.5, 2.5)

        if (self.c1 + self.c2) > 4:
            sum_c = self.c1 + self.c2
            self.c1 = (self.c1 / sum_c) * 4
            self.c2 = (self.c2 / sum_c) * 4

    def _elitist_learning(
        self, function: Callable, current_iteration: int, n_iterations: int
    ) -> None:
        p_position = self.gbest_position.copy()

        sigma_max = 1.0
        sigma_min = 0.1

        sigma = sigma_max - (sigma_max - sigma_min) * (current_iteration / n_iterations)

        d_idx = np.random.randint(0, self.dim)
        d_max = self.high
        d_min = self.low

        gaussian_noise = np.random.normal(loc=0, scale=sigma)

        p_position[d_idx] = p_position[d_idx] + (d_max - d_min) * gaussian_noise
        p_position[d_idx] = np.clip(p_position[d_idx], self.low, self.high)

        p_value = function(p_position)

        if p_value < self.gbest_value:
            self.gbest_position = p_position
            self.gbest_value = p_value
        else:
            worst_idx = np.argmax(self.current_fitness)
            self.particles_positions[worst_idx] = p_position
            self.current_fitness[worst_idx] = p_value

    def optimize(
        self, function: Callable, n_iterations: int
    ) -> Tuple[float, np.ndarray, List[float]]:
        self.history = {"fitness": [], "w": [], "c1": [], "c2": []}

        for iteration in range(n_iterations):
            self.current_fitness = np.apply_along_axis(
                function, axis=1, arr=self.particles_positions
            )

            mask_better = self.current_fitness < self.pbest_values

            self.pbest_values[mask_better] = self.current_fitness[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(self.pbest_values)

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = self.pbest_values[min_idx]
                self.gbest_position = self.pbest_positions[min_idx].copy()

            f = self._calculate_evolucionary_factor()
            state = self._get_state(f)

            self.previous_state = self.current_state
            self.current_state = state

            self._adapt_parameters(state=state, f=f)

            if state == "convergence":
                self._elitist_learning(function, iteration, n_iterations)

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = self.w * self.particles_velocities
            cognitive = self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = self.c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            self.history["fitness"].append(self.gbest_value)
            self.history["w"].append(self.w)
            self.history["c1"].append(self.c1)
            self.history["c2"].append(self.c2)

        return self.gbest_value, self.gbest_position, self.history

    def __repr__(self):
        return f"APSO(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class UAPSO:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_min: float = 0.0,
        w_max: float = 1.0,
        c1_min: float = 0.0,
        c1_max: float = 4.0,
        c2_min: float = 0.0,
        c2_max: float = 4.0,
        threshold: float = 0.5,
        learning_rate: float = 0.01,
        dim: int = 30,
    ):
        self.n_particles = n_particles
        self.dim = dim
        self.low = low
        self.high = high
        self.threshold = threshold
        self.learning_rate = learning_rate

        self.la_w_actions = np.linspace(w_min, w_max, 20)
        self.la_c1_actions = np.linspace(c1_min, c1_max, 10)
        self.la_c2_actions = np.linspace(c2_min, c2_max, 10)

        self.probs_la_w = np.full(20, 1.0 / 20)
        self.probs_la_c1 = np.full(10, 1.0 / 10)
        self.probs_la_c2 = np.full(10, 1.0 / 10)

        self.v_max = 0.2 * (high - low)
        self.particles_positions = np.random.uniform(low, high, (n_particles, dim))
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)
        self.history = {"fitness": [], "w": [], "c1": [], "c2": []}

    def _update_probs(
        self, probs: np.ndarray, chosen_idx: int, is_success: bool
    ) -> np.ndarray:
        r = len(probs)
        a = self.learning_rate
        b = self.learning_rate

        new_probs = probs.copy()

        if is_success:
            new_probs[chosen_idx] += a * (1 - new_probs[chosen_idx])

            mask = np.arange(r) != chosen_idx
            new_probs[mask] *= 1 - a
        else:
            new_probs[chosen_idx] *= 1 - b

            mask = np.arange(r) != chosen_idx
            dist_term = b / (r - 1)
            new_probs[mask] = dist_term + (1 - b) * new_probs[mask]

        return new_probs / new_probs.sum()

    def optimize(
        self, function: Callable, n_iterations: int
    ) -> Tuple[float, np.ndarray, dict]:
        self.current_fitness = np.apply_along_axis(
            function, 1, self.particles_positions
        )
        self.pbest_values = self.current_fitness.copy()
        self.pbest_positions = self.particles_positions.copy()

        min_idx = np.argmin(self.current_fitness)
        if self.current_fitness[min_idx] < self.gbest_value:
            self.gbest_value = self.current_fitness[min_idx]
            self.gbest_position = self.particles_positions[min_idx].copy()

        for iteration in range(n_iterations):
            previous_fitness = self.current_fitness.copy()

            idx_w = np.random.choice(len(self.la_w_actions), p=self.probs_la_w)
            idx_c1 = np.random.choice(len(self.la_c1_actions), p=self.probs_la_c1)
            idx_c2 = np.random.choice(len(self.la_c2_actions), p=self.probs_la_c2)

            w = self.la_w_actions[idx_w]
            c1 = self.la_c1_actions[idx_c1]
            c2 = self.la_c2_actions[idx_c2]

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            inertia = w * self.particles_velocities
            cognitive = c1 * r1 * (self.pbest_positions - self.particles_positions)
            social = c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social

            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities
            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            self.current_fitness = np.apply_along_axis(
                function, 1, self.particles_positions
            )

            mask_better = self.current_fitness < self.pbest_values
            self.pbest_values[mask_better] = self.current_fitness[mask_better]
            self.pbest_positions[mask_better] = self.particles_positions[mask_better]

            min_idx = np.argmin(self.pbest_values)
            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = self.pbest_values[min_idx]
                self.gbest_position = self.pbest_positions[min_idx].copy()

            n_improved = np.sum(self.current_fitness < previous_fitness)
            ratio = n_improved / self.n_particles

            is_successful = ratio >= self.threshold

            self.probs_la_w = self._update_probs(self.probs_la_w, idx_w, is_successful)
            self.probs_la_c1 = self._update_probs(
                self.probs_la_c1, idx_c1, is_successful
            )
            self.probs_la_c2 = self._update_probs(
                self.probs_la_c2, idx_c2, is_successful
            )

            self.history["fitness"].append(self.gbest_value)
            self.history["w"].append(w)
            self.history["c1"].append(c1)
            self.history["c2"].append(c2)

        return self.gbest_value, self.gbest_position, self.history

    def __repr__(self):
        return f"UAPSO(n_particles={self.n_particles}, dim={self.dim}, current_gbest={self.gbest_value:.4f})"
