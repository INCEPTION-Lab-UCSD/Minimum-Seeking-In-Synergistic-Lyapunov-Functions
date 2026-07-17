from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from hybrid_solution import HybridSolution


class SO3:
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
        angular_velocity,
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
        self.angular_velocity = angular_velocity
        self.values = (-1.0, 0.0, 1.0)

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule()
        else:
            self.theta_schedule = theta_schedule

    def dynamics(self, t, y):

        pass

    def diffeomorphism(self, R):
        return R.flatten("F")

    def lyapunov_function_so3(self, R, q):
        return self.potential_function(self.synergistic_potential_function(R, q))

    def lyapunov_function(self, p, q):
        R = p.reshape((3, 3))

        return self.lyapunov_function_so3(R, q)

    def control(self, y):
        p, eta, q = y[:9], y[9:-1], y[-1]
        V = self.lyapunov_function(p, q)
        e1 = np.array([1, 0], dtype=float)

        direction = expm(-self.kappa * V * self.S) @ e1

        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.control_gain_constants * self.kappa)
        )

        return gain * float(np.dot(direction, eta))

    def potential_function(self, R):
        angular_velocity_sum = np.sum(np.array(np.eye(3) @ self.angular_velocity))

        A = (
            3
            / angular_velocity_sum
            * np.array(
                [self.angular_velocity[i] for i in range(len(self.angular_velocity))]
            )
        )

        return np.linalg.trace(A @ (np.eye(3) - R))

    def synergistic_potential_function(self, R, q):
        angular_velocity_unit = self.angular_velocity / np.linalg.norm(
            self.angular_velocity
        )
        angular_velocity_change = self._skew_symmetric(angular_velocity_unit)

    def synergy_gap(self):

        pass

    def jump_map(self, y):
        p = y[:9]
        y[-1] = self.argmin_mode(p)
        return y
        

    def control_vector_fields(self, p, idx):
        e_i = np.eye(3)[idx]

        e_i_skew_symmetric = self._skew_symmetric(e_i)

        return  - np.outer(e_i_skew_symmetric, np.eye(3)) @ p

    def argmin_mode(self, p):
        return np.argmin(np.array([self.lyapunov_function(p, q) for q in self.Q])) + 1

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
                theta, values, required_duration, monitor, chi_1, chi_2
            )
            theta = tuple(candidates[rng.integers(len(candidates))])
            schedule.append((t, theta))

        return schedule

    def _apply_jump(self, y):
        p = y[:9]
        q = y[-1]
        if self.synergy_gap(p, q) >= self.delta:
            return self.jump_map(y)
        return y

    def _theta_has_zero(self, theta):
        return True if 0.0 in theta else False

    def _theta_candidates(self):


    def _skew_symmetric(self, arr):
        a, b, c = arr[0], arr[1], arr[2]
        return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]])
