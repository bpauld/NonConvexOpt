import numpy as np
import scipy as sp

def solve_boyd(prob, opt_value_bidual, verbose=True):
    m = prob.m
    n = prob.n
    A = prob.A
    b = prob.b
    b_bar = prob.b_bar

    w = np.random.randn(n)
    w = w / np.linalg.norm(w)

    def f(x):
        return w.dot(x)

    constraint1 = sp.optimize.LinearConstraint(prob.A, lb=-np.inf, ub=prob.b)
    constraint2 = sp.optimize.LinearConstraint(np.eye(n), lb=np.zeros(n), ub=np.ones(n))
    constraint3 = sp.optimize.NonlinearConstraint(prob.f_biconjugate, lb=-np.inf, ub=opt_value_bidual)
    x_start = np.zeros(n)
    res = sp.optimize.minimize(f, x_start, constraints=[constraint1, constraint2, constraint3], tol=1e-9)
    return res.x