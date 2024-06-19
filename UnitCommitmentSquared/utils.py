import numpy as np
from unit_commitment import UnitCommitment



def create_uc_problem(n, N, random_seed=None,  rho_factor=1):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    D = np.random.uniform(low=50, high=300, size=N)

    avg_gen = 1/n * np.random.uniform(low=100, high=300, size=n)
    max_gen = 2 * avg_gen
    min_gen = 0.5 * avg_gen

    assert np.sum(max_gen) > np.max(D)

    beta = np.random.uniform(low=1, high=20, size=n)
    gamma = np.random.uniform(low=3, high=5, size=n)
    omega = np.random.uniform(low=30, high=50, size=n)

    c01 = 0.5 * np.ones(n) * np.mean(beta * avg_gen**2 + gamma * avg_gen + omega)
    c10 = 0.25 * c01
    rho = rho_factor * np.max(max_gen)

    return UnitCommitment(n, N, D, beta, gamma, omega, c01, c10, min_gen, max_gen, rho=rho)