import numpy as np
from pev import *


def create_pevs_problem(n, N, random_seed=0):
    np.random.seed(random_seed)
    delta_T = 1/3
    xi = np.random.uniform(low=0.015, high=0.075, size=n)
    xi_u = 1 - xi
    xi_v = 1 + xi #useless for charging only scenario
    P = np.random.uniform(low=3, high=5, size=n)
    E_min = np.ones(n)
    E_max = np.random.uniform(low=8, high=16, size=n)
    E_init = np.random.uniform(low=0.2, high=0.5) * E_max
    E_ref = np.random.uniform(low=0.55, high=0.8) * E_max
    P_max = 3 * n * np.ones(N)
    P_min = - P_max  # no need to consider P_min in charging only scenario if P_min < 0
    Cu = np.random.uniform(low=19, high=35, size=N)
    #Cv = 1.1 * Cu

    delta_u = np.random.uniform(low=-0.3, high=0.3, size=n)
    delta_v = np.random.uniform(low=-0.3, high=0.3, size=n)

    #compute rho
    rho = N * np.max(P)
    P_max_bar = P_max - rho
    P_min_bar = P_min + rho

    return PEVProblem(n, N, delta_T, xi_u, xi_v, P, E_min, E_max, E_init, E_ref, P_max, Cu, rho, P_max_bar, delta_u)