import sys
sys.path.insert(1, '../code/')
from non_convex_problem import NonConvexProblem
import numpy as np
import scipy as sp

class PEVProblem(NonConvexProblem):
    def __init__(self, n, m, delta_T, xi_u, xi_v, P, E_min, E_max, E_init, E_ref, P_max, Cu, rho, P_max_bar, delta_u):
        self.n = n
        self.m = m
        self.di = m
        self.delta_T = delta_T
        self.xi_u = xi_u
        self.xi_v = xi_v
        self.P = P
        self.E_min = E_min
        self.E_max = E_max
        self.E_init = E_init
        self.E_ref = E_ref
        self.P_max = P_max
        self.Cu = Cu
        self.rho = rho
        self.P_max_bar = P_max_bar
        self.b = P_max
        self.b_bar = P_max_bar
        self.delta_u = delta_u

    def get_linear_coefficient(self, i):
        return self.P[i] * (self.Cu + self.delta_u[i])

    def get_di(self, i):
        return self.di

    def get_y_ik(self, i, y_k):
        return y_k[:, i]

    def construct_A_matrix(self):
        return

    def construct_Ai_matrix(self, i):
        return

    def get_feasible_point(self):
        x = np.zeros((self.di, self.n)) #this is NOT feasible
        for i in range(self.n):
            x[:, i] = self.compute_linear_min_dom_fi(i, np.random.randn(self.di))
        return x

    def f_i(self, i, u, check_feasibility=True, check_integer_constraint=False):
        if check_feasibility:
            self.is_feasible(i, u, check_integer_constraint=check_integer_constraint)
    
        return self.P[i] * (self.Cu + self.delta_u[i]).dot(u)    
    
    def f(self, u, check_feasibility=False, check_integer_constraint=False):
        res = 0
        for i in range(self.n):
            res += self.f_i(i, u[:, i], check_feasibility=check_feasibility, check_integer_constraint=check_integer_constraint)
        return res

    
                       
    def is_feasible(self, i, u, check_integer_constraint, tol=1e-6):
        ek = self.E_init[i]
        for k in range(self.m):
            ek = ek + self.P[i] * self.delta_T * self.xi_u[i] * u[k] 
            assert ek >= self.E_min[i]
            assert ek <= self.E_max[i]
            if check_integer_constraint:
                assert u[k] * (1 - u[k]) == 0
        assert ek >= self.E_ref[i] - tol
        return True


    def f_conjugate_i(self, i, x):
        C = self.P[i] * (self.Cu + self.delta_u[i])
        
        nb_of_positive_u_needed = int(np.ceil((self.E_ref[i] - self.E_init[i]) /  (self.P[i] * self.delta_T * self.xi_u[i])))
        #print(nb_of_positive_u_needed)
        sorted_indices = np.argsort(x - C)[::-1]
        res = np.zeros_like(x)
        res[sorted_indices[:nb_of_positive_u_needed]] = 1
    
        current_energy_consumed = self.E_init[i] + nb_of_positive_u_needed * self.P[i] * self.delta_T * self.xi_u[i]
        added_indices_pointer = nb_of_positive_u_needed
        while added_indices_pointer < x.shape[0] and (x-C)[sorted_indices[added_indices_pointer]] > 0 and current_energy_consumed < self.E_max[i] - self.P[i] * self.delta_T * self.xi_u[i]:
            res[sorted_indices[added_indices_pointer]] = 1
            added_indices_pointer += 1
            current_energy_consumed += self.P[i] * self.delta_T * self.xi_u[i]
        return res.dot(x - C), res   
    
    def f_conjugate(self, x):
        for i in range(self.n):
            res += self.f_conjugate_i(i, x[:, i])[0]
        return res

    def f_biconjugate_i(self, i, x):
        return self.f_i(i, x)
    

    def compute_Ai_dot_y(self, i, y):
        return self.P[i] * y

    def compute_A_dot_x(self, x):
        if x.ndim == 1:
            x = x.reshape((self.di, self.n), order='F')
        return np.sum(x * self.P[None, :], axis=1)

    def compute_AiT_dot_g(self, i, g):
        return self.P[i] * g

    def compute_linear_min_dom_fi(self, i, z):
        #computes the minimizer of z.dot(x) over x \in dom(f_i)
    
        nb_of_positive_u_needed = int(np.ceil((self.E_ref[i] - self.E_init[i]) /  (self.P[i] * self.delta_T * self.xi_u[i])))
        sorted_indices = np.argsort(z)
        res = np.zeros_like(z)
        res[sorted_indices[:nb_of_positive_u_needed]] = 1
    
        current_energy_consumed = self.E_init[i] + nb_of_positive_u_needed * self.P[i] * self.delta_T * self.xi_u[i]
        added_indices_pointer = nb_of_positive_u_needed
        while added_indices_pointer < z.shape[0] and z[sorted_indices[added_indices_pointer]] < 0 and current_energy_consumed < self.E_max[i] - self.P[i] * self.delta_T * self.xi_u[i]:
            res[sorted_indices[added_indices_pointer]] = 1
            added_indices_pointer += 1
            current_energy_consumed += self.P[i] * self.delta_T * self.xi_u[i]
        return res

    def lmo_1(self, alpha, g):
        y_k = np.zeros((self.di, self.n))
        for i in range(self.n):
            y_k[:, i] = self.lmo_1_i(i, alpha, g)
        return y_k
        
    def lmo_1_i(self, i, alpha, g):
        AiT_g = self.compute_AiT_dot_g(i, g)
        if alpha == 0:
            y_ik = self.compute_linear_min_dom_fi(i, AiT_g)
            return y_ik
        elif alpha > 0:
            f_conjugate_y_ik, y_ik = self.f_conjugate_i(i, -AiT_g/alpha)
            return y_ik 