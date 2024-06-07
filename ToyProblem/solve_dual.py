import numpy as np
import scipy as sp

def solve_bidual(prob, verbose=True):
    m = prob.m
    n = prob.n
    A = prob.A
    b = prob.b
    b_bar = prob.b_bar

    constraint1 = sp.optimize.LinearConstraint(prob.A, lb=-np.inf, ub=prob.b)
    constraint2 = sp.optimize.LinearConstraint(np.eye(n), lb=np.zeros(n), ub=np.ones(n))
    x_start = np.zeros(n)
    res = sp.optimize.minimize(prob.f_biconjugate, x_start, constraints=[constraint1, constraint2], tol=1e-9)
    return res.x