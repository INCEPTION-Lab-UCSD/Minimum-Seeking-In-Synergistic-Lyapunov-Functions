import math

import matplotlib.pyplot as plt
import numpy as np

import hybrid_solution
import obstacle_avoidance
import sphere_2d


def run_sphere_2d():
    p0 = np.array([0, -1])
    eta0 = np.array([1, 0])
    q0 = 1

    gamma = np.sqrt(1)
    kappa = 4
    epsilon = 1 / np.sqrt(4 * np.pi)
    delta = 0.25
    t_1 = 0.0
    t_2 = 25.0
    p_target = np.array([0, 1])
    simulation = sphere_2d.Sphere_2D(
        p0, q0, p_target, gamma, kappa, epsilon, t_1, t_2, delta=delta, eta0=eta0
    )
    solution = simulation.solve()

    simulation.plot(solution)

    plt.show()


def run_obstacle_avoidance():
    z0 = np.array([0, -4], dtype=float)
    target = np.array([0, 2], dtype=float)
    q0 = 1
    eta0 = np.array([1, 0], dtype=float)
    obstacle_radius = 1
    obstacle_center = np.array([0, 0], dtype=float)
    epsilon = 1 / np.sqrt(6 * np.pi)
    gamma = 2.0
    chi_1 = 1.0
    chi_2 = 0.5
    t_1 = 0.0
    t_2 = 40.0
    delta = 0.25
    kappa = 4.0
    T_0 = 1.0

    simulation = obstacle_avoidance.Target_Seeking(
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
        T_0,
        obstacle_center=obstacle_center,
        theta_seed=0,
    )

    print(simulation.theta_schedule)

    solution = simulation.solve()
    final_p = solution(t_2)[:3]
    print(simulation.diffeomorphism_inverse(final_p))


if __name__ == "__main__":
    # run_sphere_2d()
    run_obstacle_avoidance()
