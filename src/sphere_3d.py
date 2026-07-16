from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from hybrid_solution import HybridSolution


class Sphere_3D:
    Q = {1, 2}
    e1 = np.array([1, 0, 0], dtype=float)
    e2 = np.array([0, 1, 0], dtype=float)
    e3 = np.array([0, 0, 1], dtype=float)
    S = np.array([[0, 1], [-1, 0]], dtype=float)

    def __init__(
        self,
        p0,
        eta0,
        q0,
        target,
        gamma,
        delta,
        kappa,
        epsilon,
        t_1,
        t_2,
        chi_1,
        chi_2,
        control_gain_constants,
        T_0=1.0,
        theta_schedule=None,
        theta_seed=None,
    ):

        self.p0 = p0
        self.eta0 = eta0
        self.q0 = q0
        self.target = target
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.epsilon = epsilon
        self.t_1 = t_1
        self.t_2 = t_2
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.T_0 = T_0
        self.control_gain_constants = control_gain_constants
        self.values = (-1.0, 0.0, 1.0)

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule()
        else:
            self.theta_schedule = theta_schedule

    def solve(self, t=None, t_end=None, rtol=1e-6, atol=1e-9):
        if t is None:
            t = self.t_1
        if t_end is None:
            t_end = self.t_2

        y = np.r_[self.p0, self.eta0, self.q0]
        solution_segments = []

        while t < t_end - 1e-12:
            y = self._apply_jump(y)

            def q_jump_event(_, event_y):
                p = event_y[:3]
                eta = event_y[3:-1]
                q = event_y[-1]
                return self.delta - self.synergy_gap(p, q)

            q_jump_event.termainl = True
            q_jump_event.direction = -1

            sol = solve_ivp(
                fun=self.dynamics,
                t_span=(t, t_end),
                y0=y,
                method="RK45",
                rtol=rtol,
                atol=atol,
                dense_ouput=True,
                events=q_jump_event,
            )

            solution_segments.append(sol)

            if sol.status != 1:
                break

            t = float(sol.t[-1])
            y = self.jump_map(sol.y[:, -1])

    def dynamics(self, t, y):
        p = y[:3]
        eta = y[3:-1]
        q = y[-1]
        p_dot = np.empty_like(p)
        for idx, theta in enumerate(self.get_control_gain(t)):
            p_dot += self.control_vector_fields(p, idx) * theta

        eta_dot = np.empty_like(eta)
        eta_dot = (
            2.0
            * np.pi
            * self.control_gain_constants**-1
            * self.epsilon**-2
            * (self.S @ eta)
        )

        for idx, T_i in enumerate(self.control_gain_constants):
            eta_dot[idx] = 2.0 * np.pi * T_i**-1 * self.epsilon**-2 * (self.S @ eta)

        return np.r_[p_dot, eta_dot, q]

    def get_control_gain(self, t):
        for i in range(len(self.theta_schedule) - 1):
            segment_start_time = self.theta_schedule[i][0]
            next_segment_start_time = self.theta_schedule[i + 1][0]
            if t >= segment_start_time and t < next_segment_start_time:
                return self.theta_schedule[i][1]

        ValueError("t is not in range")

    def lyapunov_function(self, p, q):
        return self.potential_function(self.synergistic_potential_function(p, q))

    def potential_function(self, p):
        return 1 - np.dot(p, self.target)

    def synergistic_potential_function(self, p, q):
        target_orthogonal = (
            np.cross(self.target, self.e1)
            if np.allclose(self.target, self.e2)
            else np.cross(self.target, self.e2)
        )

        return (
            expm(
                (3 / 2 - q)
                * self.potential_function(p)
                @ self._skew_symmetric(target_orthogonal)
            )
            @ p
        )

    def control_vector_fields(self, p, idx):
        e_i = np.eye(3)[idx]
        return e_i - np.dot(p, e_i) * p

    def control(self, p, q, eta):
        V = self.lyapunov_function(p, q)
        e1 = np.array([1, 0], dtype=float)

        direction = expm(-self.kappa * V * self.S) @ e1

        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.control_gain_constants * self.kappa)
        )

        return gain * float(np.dot(direction, eta))

    def synergy_gap(self, p, q):
        values = [self.lyapunov_function(p, q_i) for q_i in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def jump_map(self, y):
        y_plus = np.array(y, dtype=float).copy()
        p = y_plus[:3]
        y_plus[-1] = self.argmin_mode(p)

        return y_plus

    def argmin_mode(self, p):
        values = np.array([self.lyapunov_function(p, q) for q in self.Q])
        return int(np.argmin(values)) + 1

    def generate_theta_schedule(self, initial_theta=(1.0, 1.0, 1.0), seed=None):
        values = self.values
        t_start = self.t_1
        t_end = self.t_2
        chi_1 = self.chi_1
        chi_2 = self.chi_2
        T_0 = self.T_0

        theta = initial_theta

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

    def _apply_jump(self, y):
        p = y[:3]
        q = y[-1]
        if self.synergy_gap(p, q) >= self.delta:
            return self.jump_map(y)
        return y

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

    def _theta_has_zero(self, theta):
        return np.any(np.isclose(value, 0.0) for value in theta)

    def _skew_symmetric(self, arr):
        a, b, c = arr[0], arr[1], arr[2]
        return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]])
