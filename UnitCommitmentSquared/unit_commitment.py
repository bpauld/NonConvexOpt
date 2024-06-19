import sys
sys.path.insert(1, '/home/bpauldub/ShapleyFolkman/code/')
from non_convex_problem import NonConvexProblem
import numpy as np


class UnitCommitment(NonConvexProblem):
    def __init__(self, n, N, D, beta, gamma, omega, c01, c10, min_gen, max_gen, rho=0):
        self.n = n
        self.m = N
        self.N = N
        self.di = 2 * N
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.c01 = c01
        self.c10 = c10
        self.min_gen = min_gen
        self.max_gen = max_gen
        self.D = D
        self.b = -D
        self.rho = rho
        self.b_bar = self.b - self.rho

    def construct_A_matrix(self):
        A = np.zeros((self.m, 2*self.N*self.n))
        for i in range(self.n):
            A[:, 2*i*self.N:2*(i+1)*self.N] = self.construct_Ai_matrix(i)
        return A

    def construct_Ai_matrix(self, i):
        Ai = np.zeros((self.m, 2*self.N))
        Ai[:, self.N:] = np.eye(self.N)
        return -Ai

    def check_domain_i(self, i, x, unfeasibility_tol=1e-6):
        u = x[:self.N] #state part
        g = x[self.N:] #power part
        assert not np.any(u * (1-u) > unfeasibility_tol) #check if state is either 0 or 1
        for t in range(self.N):
            if u[t] == 1:
                assert g[t] >= self.min_gen[i] - unfeasibility_tol
                assert g[t] <= self.max_gen[i] + unfeasibility_tol
            else:
                assert np.abs(g[t]) < unfeasibility_tol

    def f_i(self, i, x):
        u = x[:self.N] #state part
        g = x[self.N:] #power part
    
        self.check_domain_i(i, x)
        fi = 0
        #do the first step
        if u[0] == 1:
            fi += self.c01[i] + self.beta[i] * g[0]**2 + + self.gamma[i] * g[0] + self.omega[i]
    
        for t in range(1, self.N):
            if u[t-1] == u[t]:
                if u[t] == 1:
                    fi += self.beta[i] * g[t]**2 + self.gamma[i] * g[t] +  self.omega[i]
            else:
                if u[t] == 1:
                    fi += self.c01[i] + self.beta[i] * g[t]**2 + self.gamma[i] * g[t] +  self.omega[i]
                else:
                    fi += self.c10[i]
        return fi


    def f(self, x):
        if x.ndim == 1:
            x = x.reshape((self.di, self.n), order='F')
        res = 0
        for i in range(self.n):
            res += self.f_i(i, x[:, i])
        return res

    def f_biconjugate_i(self, i, x):
        return self.f_i(i, x)


    def small_quadratic_i(self, i, g, p):
        return p * g - self.beta[i] * g**2  -self.gamma[i] * g - self.omega[i]
    def small_quadratic_optimizer_i(self, i, p):
        return (p - self.gamma[i]) / (2 * self.beta[i])

    def f_conjugate_i(self, i, z):
        v = z[:self.N] #state part
        p = z[self.N:] #power part
        

        #Initialize Delta_t_0
        Delta_t_0 = 0.0
        path_state_0 = [0.0] 
        path_prod_0 = [0.0]

        #Initialize Delta_t_1
        Delta_t_1 = 0.0
        path_state_1 = [1.0]
        opt_prod_t = self.small_quadratic_optimizer_i(i, p[0])
        path_prod_1 = []
        if opt_prod_t < self.min_gen[i]:
            Delta_t_1 = v[0] + self.small_quadratic_i(i, self.min_gen[i], p[0]) - self.c01[i]
            path_prod_1.append(self.min_gen[i])
        elif opt_prod_t > self.max_gen[i]:
            Delta_t_1 = v[0] + self.small_quadratic_i(i, self.max_gen[i], p[0]) - self.c01[i]
            path_prod_1.append(self.max_gen[i])
        else:
            Delta_t_1 = v[0] + self.small_quadratic_i(i, opt_prod_t, p[0]) - self.c01[i]
            path_prod_1.append(opt_prod_t)
        
        
        for t in range(1, self.N):
            Delta_t1_0 = max(Delta_t_0, Delta_t_1 - self.c10[i]) 
            
            Delta_t1_1 = 0
            opt_prod_t = self.small_quadratic_optimizer_i(i, p[t])
            final_prod_t = - np.inf
            if opt_prod_t < self.min_gen[i]:
                Delta_t1_1 = v[t] + self.small_quadratic_i(i, self.min_gen[i], p[t])
                final_prod_t = self.min_gen[i]
            elif opt_prod_t > self.max_gen[i]:
                Delta_t1_1 = v[t] + self.small_quadratic_i(i, self.max_gen[i], p[t])
                final_prod_t = self.max_gen[i]
            else:
                Delta_t1_1 = v[t] + self.small_quadratic_i(i, opt_prod_t, p[t])
                final_prod_t = opt_prod_t
            Delta_t1_1 += max(Delta_t_1, Delta_t_0 - self.c01[i])
        
            if Delta_t1_0 == Delta_t_0:
                next_path_state_0 = path_state_0 + [0.0]
                next_path_prod_0 = path_prod_0 + [0.0]
            else:
                next_path_state_0 = path_state_1 + [0.0]
                next_path_prod_0 = path_prod_1 + [0.0]


            if Delta_t_1 > Delta_t_0 - self.c01[i]:
                next_path_state_1 = path_state_1 + [1.0]
                next_path_prod_1 = path_prod_1 + [final_prod_t]
            else:
                next_path_state_1 = path_state_0 + [1.0]
                next_path_prod_1 = path_prod_0 + [final_prod_t]
        
            path_state_0 = next_path_state_0
            path_prod_0 = next_path_prod_0
            path_state_1 = next_path_state_1
            path_prod_1 = next_path_prod_1
            Delta_t_0 = Delta_t1_0
            Delta_t_1 = Delta_t1_1
        
        Delta_t = max(Delta_t_0, Delta_t_1)
        last_opt = 0
        if Delta_t == Delta_t_0:
            opt_path_state = path_state_0
            opt_path_prod = path_prod_0
            last_opt = 0
        else:
            opt_path_state = path_state_1
            opt_path_prod = path_prod_1
            last_opt = 1
        #print(len(opt_path_state), len(opt_path_prod))
        grad = np.array(opt_path_state + opt_path_prod)
        return Delta_t, grad


    def compute_linear_min_dom_fi(self, i, z):
        #computes the minimizer of z.dot(x) over dom f_i
        N = self.N
        v = z[:N] #state part
        p = z[N:] #power part
        x = np.zeros(2*N)
    
        candidates = np.zeros((N, 3))
        candidates[:, 1] = v + p * self.min_gen[i]
        candidates[:, 2] = v + p * self.max_gen[i]
        minimizers = np.argmin(candidates, axis=1)
        x[:N] = np.ones(N) * (2 - minimizers) * minimizers  + np.ones(N) * (1 - minimizers) * minimizers * (-0.5)
        x[N:] = self.min_gen[i] * (2 - minimizers) * minimizers + self.max_gen[i] * (1 - minimizers) * minimizers * (-0.5)
        return x


    def lmo_1(self, i, alpha, g):
        if alpha == 0:
            y_ik = self.compute_linear_min_dom_fi(i, g)
            return y_ik
        elif alpha > 0:
            f_conjugate_y_ik, y_ik = self.f_conjugate_i(i, -g/alpha)
            return y_ik 

    def get_feasible_point(self):
        x_feasible = np.ones((self.di, self.n))
        for i in range(self.n):
            x_feasible[self.N:, i] = self.max_gen[i]
        return x_feasible

    def build_solution_in_domain(self, y_final):
        y_final_in_domain = y_final.copy()
        for i in range(self.n):
            for t in range(self.N):
                if y_final[t, i] > 1 and y_final[t, i] < 1 + 1e-6:
                    y_final_in_domain[t, i] = 1
                if y_final[t, i] > 1e-6 and y_final[t, i] < 1:
                    y_final_in_domain[t, i] = 1
                    if y_final[t+self.N, i] < self.min_gen[i]:
                        y_final_in_domain[t + self.N, i] = self.min_gen[i]
    
        return y_final_in_domain