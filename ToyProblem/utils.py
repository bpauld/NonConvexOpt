import numpy as np


def create_toy_problem(n, m, rho, random_seed=0):
    np.random.seed(random_seed)
    A = 10 * np.random.randn(m, n)
    b = A @ np.ones(n) * 0.5
    rho = rho * np.ones(m)
    b_bar = b - rho

    return ToyProblem(n, m, A, b, b_bar, rho)

class ToyProblem:
    def __init__(self, n, m, A, b, b_bar, rho, linear_by_part=False):
        self.n = n
        self.m = m
        self.di = 1
        self.A = A
        self.b = b
        self.b_bar = b_bar
        self.rho = rho
        self.linear_by_part = linear_by_part

    def f_i(self, i, x):
        if linear_by_part:
            slope1 = -10
            slope2 = 0
            slope3 = 10
            c1 = 5
            c2 = 1
            c3 = -5
            if np.isscalar(x):
                return max(slope1 * x + c1, slope2 * x + c2, slope3 * x + c3)
            else:
                return max(slope1 * x[0] + c1, slope2 * x[0] + c2, slope3 * x[0] + c3)
        else:
            if x <= 0.4 or x >= 0.6:
                if np.isscalar(x):
                    return 100*(x-0.5)**2
                else:
                    return 100*(x[0]-0.5)**2
            else:
                return 10

    def f(self, x):
        if x.ndim == 1:
            x = x.reshape((self.di, self.n), order='F')
        res = 0
        for i in range(self.n):
            res += self.f_i(i, x[:, i])
        return res

    def f_biconjugate_i(self, i, x):
        if x < 0.4 or x>0.6:
            return self.f_i(i, x)
        else:
            return 1
    
    def f_biconjugate(self, x):
        if x.ndim == 1:
            x = x.reshape((self.di, self.n), order='F')
        res = 0
        for i in range(self.n):
            res += self.f_biconjugate_i(i, x[:, i])
        return res
    
    def f_conjugate_i(self, i, v):
        opt_quadratic = v/200 + 0.5
        if opt_quadratic < 0:
            opt_quadratic = 0
        elif opt_quadratic > 1:
            opt_quadratic = 1
        elif opt_quadratic > 0.4 and opt_quadratic < 0.6:
            if v * 0.4 - self.f_i(i, 0.4) > v * 0.6 - self.f_i(i, 0.6):
                opt_quadratic = 0.4
            else:
                opt_quadratic = 0.6

        opt_linear = 0.6 if v > 0 else 0.4

        if v * opt_linear - 1 > v * opt_quadratic - self.f_i(i, opt_quadratic):
            return v*opt_linear, np.ones(1) * opt_linear
        else:
            return v * opt_quadratic - self.f_i(i, opt_quadratic), np.ones(1) * opt_quadratic
                
    def construct_Ai_matrix(self, i):
        return self.A[:, i].reshape((self.m, 1))
    
    def construct_A_matrix(self):
        return self.A


    def compute_linear_min_dom_fi(self, i, z):
        #computes the minimizer of z.dot(x) over dom f_i
        if z > 0:
            return np.zeros(1)
        else:
            return np.ones(1)

    def lmo_1(self, i, alpha, g):
        if alpha == 0:
            y_ik = self.compute_linear_min_dom_fi(i, g)
            return y_ik
        elif alpha > 0:
            f_conjugate_y_ik, y_ik = self.f_conjugate_i(i, -g/alpha)
            return y_ik 