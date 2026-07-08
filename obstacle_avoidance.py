from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import sympy
import torch
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
        eta0,
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
        T_1,
        theta_schedule=None,
        obstacle_center=np.array([0.0, 0.0], dtype=float),
        theta_seed=None,
    ):

        self.q0 = q0
        self.eta0 = eta0
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
        self.T_1 = T_1
        self.obstacle_center = obstacle_center

        self.p0 = self.diffeomorphism(z0)

        self.p_target = self.diffeomorphism(target)

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule()
        else:
            self.theta_schedule = theta_schedule

    def control_gain(self, t):
        return np.array([0.0, 1.0])
        if callable(self.theta_schedule):
            return float(self.theta_schedule(t))

        if self.theta_schedule is not None:
            value = self.theta_schedule[0][1]
            for switch_time, candidate in self.theta_schedule:
                if t < switch_time:
                    break
                value = candidate
            return float(value)

        return self._default_control_gain(t)

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

    def _diffeomorphism_torch(self, z):
        center = torch.as_tensor(self.obstacle_center, dtype=z.dtype, device=z.device)
        diff = z - center
        norm = torch.linalg.norm(diff)
        z_unit = diff / norm
        log_term = torch.log(norm - self.obstacle_radius).unsqueeze(0)
        return torch.cat([log_term, z_unit])

    def diffeomorphism_jacobian(self, z):
        z_tensor = torch.as_tensor(z, dtype=torch.float64)

        jacobian = torch.autograd.functional.jacobian(
            self._diffeomorphism_torch, z_tensor
        )
        return jacobian.detach().cpu().numpy()

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
            + np.sqrt(np.exp(radius) - np.exp(radius_target) ** 2 + 1)
            - 1
            + self.synergistic_potential_function(vartheta, q)
        )

    def dynamics(self, t, y):
        p = y[:3]
        eta = y[3:-1]
        q = y[-1]
        b = self.control_vector_fields(p)

        theta = np.asarray(self.control_gain(t), dtype=float).reshape(-1)
        u = np.asarray(self.control(p, q, eta), dtype=float)
        if u.ndim == 0:
            u = np.full(theta.shape, float(u))

        p_dot = b @ (theta * u)
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
            y = self._apply_jump_if_needed(y)

            def q_jump_event(_, event_y):
                p = self._unit(event_y[:2])
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
        # b = D gamma circ gamma_inv(p)

        # equivalent to using the unit vector approach
        b = self.diffeomorphism_jacobian(self.diffeomorphism_inverse(p)) @ np.eye(2)
        return b @ np.eye(b.shape[1])

    def synergy_gap(self, p, q):
        values = [self.lyapunov_function(p, q_i) for q_i in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def generate_theta_schedule(self):
        return None

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

    def _jump_check(self, p, q):
        V = self.lyapunov_function(p, q)
        V_min = np.min(
            np.array([self.lyapunov_function(p, mode) for mode in self.Q], dtype=float)
        )

        return

    def _mode(self, y):
        q = y[-1]
        return int(round(q))
