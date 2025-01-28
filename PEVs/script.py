import os

# Set the environment variable
os.environ['SCIPY_USE_PROPACK'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/bpauldub/ShapleyFolkman/')
sys.path.insert(1, '/home/bpauldub/ShapleyFolkman/code/')

from frank_wolfe_1 import *
from approximate_caratheodory import sparsify_solution_approximate_caratheodory_fcfw, compute_f_values, build_final_solution_approximate_caratheodory, sparsify_solution_approximate_caratheodory_mnp


import scipy as sp
import os
from utils import create_pevs_problem
from solve_dual_gd import solve_dual_gd
from pev import *
from solve_randomized import *



def run_exp_fixed_K(prob, K, opt_dual_value):
    x_start = prob.get_feasible_point()
    z_K, grad_norm_list, y_dic, = frank_wolfe_1(opt_dual_value, prob.P_max_bar, K, prob, x_start, verbose=True, keep_stepsizes=False)
    T = 10 * pev_prob.n # this should be enough
    z_t, active_set, opt_lambda = sparsify_solution_approximate_caratheodory_mnp(z_K, T, y_dic, pev_prob, f_values=None)
    y_final_mnp, y_final_mnp_max = build_final_solution_approximate_caratheodory(y_dic, opt_lambda, active_set)
    return y_final_mnp_max


def run_exp(prob, K_range=[50, 100, 200, 500, 1000, 5000, 10000], max_iter_gd=10000):
    scores = np.zeros((3, len(K_range) + 1))

    #Get optimal dual value of perturbed problem
    opt_dual_value = solve_dual_gd(prob, eta="1/k", max_iter=max_iter_gd, solve_contracted_problem=True, verbose=False)
    print(f"Opt dual value = {opt_dual_value}")

    #Solve using udell method, slightly increase the optimal dual value to ensure that the resulting LP is not infeasible due to numerical errors
    found_solution = -1
    perturbation = 1
    alpha = 0.001
    while found_solution==-1:
        u, found_solution = solve_randomized(perturbation*opt_dual_value, prob, solve_contracted=True)
        perturbation += alpha
    #get primal feasible solution
    u_final = build_final_solution_randomized(u, prob)
    infeasibility_udell = np.linalg.norm(np.clip(prob.compute_A_dot_x(u_final) - prob.b, 0, None))
    infeasibility_udell_bar = np.linalg.norm(np.clip(prob.compute_A_dot_x(u_final) - prob.b_bar, 0, None))
    fval_udell = prob.f(u_final, check_feasibility=True, check_integer_constraint=True)
    scores[0, -1] = fval_udell
    scores[1, -1] = infeasibility_udell
    scores[2, -1] = infeasibility_udell_bar

    #now run our method for different values of K
    for index_K, K in enumerate(K_range):
        y_final_mnp_max = run_exp_fixed_K(prob, K, opt_dual_value)
        infeasibility_K = np.linalg.norm(np.clip(prob.compute_A_dot_x(y_final_mnp_max) - prob.b, 0, None))
        infeasibility_K_bar = np.linalg.norm(np.clip(prob.compute_A_dot_x(y_final_mnp_max) - prob.b_bar, 0, None))
        fval_K = prob.f(y_final_mnp_max, check_feasibility=True, check_integer_constraint=True)
        scores[0, index_K] = fval_K
        scores[1, index_K] = infeasibility_K
        scores[2, index_K] = infeasibility_K_bar
    return scores



if __name__ == "__main__": 
    nb_experiments = 10
    n = 500
    N = 24
    m =  N
    K_range = [50, 100, 150, 200, 350, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
    max_iter_gd=10000
    final_scores = scores = np.zeros((nb_experiments, 3, len(K_range) + 1))
    for random_seed in range(nb_experiments):
        pev_prob = create_pevs_problem(n, N, random_seed=random_seed)
        scores = run_exp(pev_prob, K_range=K_range, max_iter_gd=max_iter_gd)
        final_scores[random_seed, :, :] = scores
    filename=f"results/pev_n_{n}_N_{N}_Krange_{len(K_range)}_nb_experiments_{nb_experiments}.npy"
    np.save(filename, final_scores)


