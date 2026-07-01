import matplotlib.pyplot as plt
import numpy as np
import sympy
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
    ):
        self.z0 = z0
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

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule()
        else:
            self.theta_schedule = theta_schedule

    def control_gains(self):
        pass

    def diffeomorphism(self, z):
        diff = z - self.target
        norm = np.linalg.norm(diff)
        z_unit = diff / norm
        return np.log(norm - self.obstacle_radius), z_unit

    def diffeomorphism_inverse(self, log_r, z_unit):
        z_unit = z_unit / np.linalg.norm(z_unit)
        norm = np.exp(log_r) + self.obstacle_radius
        return self.target + z_unit * norm

    def potential_function(self, p):
        return 1.0 - float(np.dot(self.target, p))

    def sphere_map(self, p, q):
        return expm((1.5 - q) * self.potential_function(p) * self.S) @ p

    def synergistic_potential_function(self, p, q):
        return self.potential_function(self.sphere_map(p, q))

    def lyapunov_function(self, p, q):
        rho = p[0]
        vartheta = p[1:]
        return (
            0.5 * (rho - self.target) ** 2
            + np.sqrt(np.exp(rho) - np.exp(self.target) ** 2 + 1)
            - 1
            + self.potential_function(vartheta)
        )

    def dynamics(self):
        pass

    def solve(self, t=None, t_end=None, rtol=1e-6, atol=1e-8):

        if t is None:
            t = self.t_1
        if t_end is None:
            t_end = self.t_2
        p0 = self.diffeomorphism(self.z0)

        # state is [p = (rho, v), eta, q]
        y = np.r_[p0, self.eta0, self.q0]
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

        pass

    def control(self, p, q, eta):
        V = self.lyapunov_function(p, q)
        e1 = np.array([1, 0, 0], dtype=float)

        direction = expm(-self.kappa * V * self.S) @ e1
        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.T_1 * self.kappa)
        )

        return gain * float(np.dot(direction, eta))

    def control_vector_fields(self, p):
        rho = p[0]
        vartheta = p[1:]
        b_rho = np.exp(-rho) * vartheta
        pass

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
