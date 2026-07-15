import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from hybrid_solution import HybridSolution


class Target_Seeking:
    S = np.array([[0, 1], [-1, 0]], dtype=float)
    Q = {1, 2}

    def __init__(
        self,
        z0,
        q0,
        oscillators,
        target,
        obstacle_radius,
        gamma,
        delta,
        kappa,
        epsilon,
        t_1,
        t_2,
        chi_1,
        chi_2,
        T_0,
        theta_schedule=None,
        obstacle_center=np.array([0.0, 0.0], dtype=float),
        theta_seed=None,
        T_1=1.0,
    ):

        self.q0 = q0
        self.eta0 = self._unit(oscillators)
        self.T_1 = float(T_1)
        self.target = target
        self.obstacle_radius = obstacle_radius
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa
        self.epsilon = epsilon
        self.t_1 = t_1
        self.t_2 = t_2
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.T_0 = T_0
        self.obstacle_center = np.asarray(obstacle_center, dtype=float)

        self.p0 = self.diffeomorphism(z0)

        self.p_target = self.diffeomorphism(target)

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule(seed=theta_seed)
        else:
            self.theta_schedule = theta_schedule

    def control_gain(self, t):

        if callable(self.theta_schedule):
            return self._input_vector(self.theta_schedule(t), "theta")

        if self.theta_schedule is not None:
            value = self.theta_schedule[0][1]
            for switch_time, candidate in self.theta_schedule:
                if t < switch_time:
                    break
                value = candidate
            return self._input_vector(value, "theta")

        return np.ones(2, dtype=float)

    def diffeomorphism(self, z):
        diff = z - self.obstacle_center
        norm = np.linalg.norm(diff)
        z_unit = diff / norm
        return np.concatenate([np.array([np.log(norm - self.obstacle_radius)]), z_unit])

    def diffeomorphism_inverse(self, p):
        p = np.asarray(p)
        radius = p[0]
        unit_vec = self._unit(p[1:])

        radius = self.obstacle_radius + np.exp(radius)
        return self.obstacle_center + radius * unit_vec

    def potential_function(self, unit_vec):
        target_unit_vec = self.p_target[1:]
        return 1.0 - float(np.dot(target_unit_vec, unit_vec))

    def sphere_map(self, unit_vec, q):

        return expm((1.5 - q) * self.potential_function(unit_vec) * self.S) @ unit_vec

    def synergistic_potential_function(self, unit_vec, q):
        return self.potential_function(self.sphere_map(unit_vec, q))

    def lyapunov_function(self, p, q):
        radius = p[0]
        radius_target = self.p_target[0]
        vartheta = p[1:]

        return (
            0.5 * (radius - radius_target) ** 2
            + np.sqrt((np.exp(radius) - np.exp(radius_target)) ** 2 + 1)
            - 1
            + self.synergistic_potential_function(vartheta, q)
        )

    def dynamics(self, t, y):
        p = y[:3]
        eta = y[3:-1]
        q = y[-1]
        b = self.control_vector_fields(p)

        theta = self.control_gain(t)
        u = self._input_vector(self.control(p, q, eta), "u")

        p_dot = np.sum(b * (theta * u), axis=1)
        eta_dot = 2.0 * np.pi * self.T_1**-1 * self.epsilon**-2 * (self.S @ eta)
        q_dot = 0.0

        return np.r_[p_dot, eta_dot, q_dot]

    def solve(self, t=None, t_end=None, rtol=1e-6, atol=1e-8):

        if t is None:
            t = self.t_1
        if t_end is None:
            t_end = self.t_2

        # state is [p = (rho, v), eta, q]
        y = np.r_[self.p0, self.eta0, self.q0]
        solution_segments = []
        while t < t_end - 1e-12:
            y = self._apply_jump(y)

            def q_jump_event(_, event_y):
                p = event_y[:3]
                q = self._mode(event_y)
                return self.delta - self.synergy_gap(p, q)

            q_jump_event.terminal = True
            q_jump_event.direction = -1

            sol = solve_ivp(
                fun=self.dynamics,
                t_span=(t, t_end),
                y0=y,
                method="RK45",
                rtol=rtol,
                atol=atol,
                dense_output=True,
                events=q_jump_event,
            )
            solution_segments.append(sol)

            if sol.status != 1:
                break

            t = float(sol.t[-1])
            y = self.jump_map(sol.y[:, -1])

        return HybridSolution(solution_segments)

    def control(self, p, q, eta):
        V = self.lyapunov_function(p, q)
        e1 = np.array([1, 0], dtype=float)

        direction = expm(-self.kappa * V * self.S) @ e1

        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.T_1 * self.kappa)
        )

        return gain * float(np.dot(direction, eta))

    def control_vector_fields(self, p):
        p = np.asarray(p, dtype=float)
        rho = p[0]
        vartheta = self._unit(p[1:])
        radius = self.obstacle_radius + np.exp(rho)

        b = np.empty((3, 2), dtype=float)
        b[0, :] = np.exp(-rho) * vartheta
        b[1:, :] = (np.eye(2) - np.outer(vartheta, vartheta)) / radius
        return b

    def synergy_gap(self, p, q):
        values = [self.lyapunov_function(p, q_i) for q_i in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def _apply_jump(self, y):
        p = y[:3]
        q = y[-1]
        if self.synergy_gap(p, q) >= self.delta:
            return self.jump_map(y)
        return y

    def generate_theta_schedule(
        self, initial_theta=(1.0, 1.0), values=(-1.0, 0.0, 1.0), seed=None
    ):
        t_start = float(self.t_1)
        t_end = float(self.t_2)
        chi_1 = float(self.chi_1)
        chi_2 = float(self.chi_2)
        T_0 = float(self.T_0)
        values = tuple(float(value) for value in values)
        theta = tuple(float(value) for value in initial_theta)

        min_dwell = 1.0 / chi_1
        schedule = [(t_start, theta)]
        t = t_start
        monitor = T_0
        rng = np.random.default_rng(seed)

        while t < t_end - 1e-12:
            remaining = t_end - t
            duration = min(min_dwell, remaining)

            if self._theta_has_zero(theta):
                duration = min(duration, monitor / (1.0 - chi_2))
                monitor -= (1.0 - chi_2) * duration
            else:
                monitor = min(T_0, monitor + chi_2 * duration)

            t += duration
            if t >= t_end - 1e-12:
                break

            required_duration = min(min_dwell, t_end - t)
            candidates = self._theta_candidates(
                theta, values, required_duration, monitor, chi_2
            )
            theta = tuple(candidates[rng.integers(len(candidates))])
            schedule.append((t, theta))

        return schedule

    @classmethod
    def _theta_candidates(cls, theta, values, required_duration, monitor, chi_2):
        candidates = [
            candidate
            for candidate in cls._theta_space(values, len(theta))
            if candidate != theta
        ]

        zero_budget_required = (1.0 - chi_2) * required_duration
        if monitor < zero_budget_required - 1e-12:
            candidates = [
                candidate
                for candidate in candidates
                if not cls._theta_has_zero(candidate)
            ]

        return candidates

    @staticmethod
    def _theta_space(values, dimension):
        if dimension == 0:
            return [()]

        tails = Target_Seeking._theta_space(values, dimension - 1)
        return [(value, *tail) for value in values for tail in tails]

    @staticmethod
    def _theta_has_zero(theta):
        return any(np.isclose(value, 0.0) for value in theta)

    def jump_map(self, y):
        y_plus = np.array(y, dtype=float).copy()
        p = y_plus[:3]
        y_plus[-1] = self.argmin_mode(p)
        return y_plus

    def argmin_mode(self, p):
        values = np.array([self.lyapunov_function(p, q) for q in self.Q])
        return int(np.argmin(values) + 1)

    def _unit(self, vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _input_vector(value, name):
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size == 1:
            return np.full(2, vector[0], dtype=float)
        if vector.size != 2:
            raise ValueError(
                f"{name} must be scalar or length 2, got shape {vector.shape}"
            )
        return vector

    def _mode(self, y):
        q = y[-1]
        return int(round(q))
