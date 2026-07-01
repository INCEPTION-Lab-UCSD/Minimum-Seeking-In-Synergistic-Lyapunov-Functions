import math

import matplotlib.pyplot as plt
import numpy as np

import hybrid_solution
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


if __name__ == "__main__":
    run_sphere_2d()
