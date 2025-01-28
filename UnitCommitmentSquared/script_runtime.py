import os

# Set the environment variable
os.environ['SCIPY_USE_PROPACK'] = '1'

import numpy as np
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../code/')

from frank_wolfe_1 import *
from frank_wolfe_1_strongly_convex import *
from exact_caratheodory import *
from approximate_caratheodory import *
from approximate_caratheodory import sparsify_solution_approximate_caratheodory_fcfw, compute_f_values, build_final_solution_approximate_caratheodory, sparsify_solution_approximate_caratheodory_mnp
from solve_dual_gd import *
import scipy as sp
import os
from utils import *
import csv
import time




def run_experiment(n_range, N=10, nb_experiments=10, K=10000, rho_factor=2, run_exact_carath=True):

    #rho_factor = 2
    #filename = f"results/results_rho_factor_{rho_factor}.csv"
    results = []


    runtimes = np.zeros((8, len(n_range), nb_experiments))


    for (index_n, n) in enumerate(n_range):
        
        for rs in range(nb_experiments):
    
            runtime_first_stage = 0
            runtime_exact_carath = 0
            runtime_approx_carath_mnp = 0
            runtime_approx_carath_fcfw = 0

        
            uc_prob = create_uc_problem(n=n, N=N, random_seed=rs, rho_factor=rho_factor)
    
            print(f"MaxMax generation = {np.max(uc_prob.max_gen)}, Sum(max_gen) - max(demand) = {np.sum(uc_prob.max_gen) - np.max(uc_prob.D)}") 
        
            start_time_first_stage = time.time()
    
            v_star_rho = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=True)
    
            max_rho_fi = 0
            for i in range(uc_prob.n):
                rho_i = uc_prob.f_i(i, np.concatenate([np.ones(uc_prob.N), np.ones(uc_prob.N) * uc_prob.max_gen[i]]))
                max_rho_fi = rho_i if rho_i > max_rho_fi else max_rho_fi
    
            #======================= Run first conditional Gradient method =======================
            x_start = uc_prob.get_feasible_point() # or initiate other feasible point
            print(v_star_rho)
            z_K, grad_norm_list, y_dic, = frank_wolfe_1(v_star_rho, uc_prob.b_bar, K, uc_prob, x_start, verbose=True)
            
            runtime_first_stage = time.time() - start_time_first_stage
    
    
            #============ Sparsify using exact Caratheodory algorithm =========================
            if run_exact_carath and n <= 200:
                start_time_exact_carath = time.time()
                Z_matrix, eta_vector, corresponding_indices = create_Z_matrix(y_dic, uc_prob)
                #eta_vector[eta_vector < 0] = 0
                #print((Z_matrix @ eta_vector)[:uc_prob.m+1] - z_K)
                print(f"Z_matrix created with shape {Z_matrix.shape}")
                final_Z_matrix, final_eta_vector, final_corresponding_indices = sparsify_solution_exact_caratheodory(Z_matrix, eta_vector, corresponding_indices, verbose=False)
    
                runtime_exact_carath = time.time() - start_time_exact_carath
    
                print("Done with exact Caratheodory ")

    
    
            #============ Sparsify using approximate Caratheodory FCFW =================
            start_time_approx_carath_fcfw = time.time()
            #f_values = compute_f_values(y_dic, uc_prob)
            T = 3 * uc_prob.n # this should be enough
            nb_non_trivial_CC_wanted = uc_prob.N
            z_t, active_set, grad_list, opt_lambda, V_matrix, _ = sparsify_solution_approximate_caratheodory_fcfw(z_K, T, y_dic, uc_prob, f_values=None,                                                                                           nb_non_trivial_CC_wanted=nb_non_trivial_CC_wanted)
            runtime_approx_carath_fcfw = time.time() - start_time_approx_carath_fcfw

            print("Done with approximate Caratheodory FCFW")

            #============ Sparsify using approximate Caratheodory <MP =================
            start_time_approx_carath_mnp = time.time()
            #f_values = compute_f_values(y_dic, uc_prob)
            T = 3 * uc_prob.n # this should be enough
            z_t, active_set, opt_lambda = sparsify_solution_approximate_caratheodory_mnp(z_K, T, y_dic, uc_prob, f_values=None)
            runtime_approx_carath_mnp = time.time() - start_time_approx_carath_mnp

            print("Done with approximate Caratheodory MNP")
        

            runtimes[0, index_n, rs] = runtime_first_stage
            runtimes[1, index_n, rs] = runtime_exact_carath
            runtimes[2, index_n, rs] = runtime_approx_carath_fcfw
            runtimes[3, index_n, rs] = runtime_approx_carath_mnp
            runtimes[4, index_n, rs] = runtime_first_stage + runtime_exact_carath
            runtimes[5, index_n, rs] = runtime_first_stage + runtime_approx_carath_fcfw
            runtimes[6, index_n, rs] = runtime_first_stage + runtime_approx_carath_mnp
            runtimes[7, index_n, rs] = n

    return runtimes

        
        

def format_decimal(x):
    return '{:.1e}'.format(x)

def format_time(x):
    return str(round(x, 2)) #only keep 2 digits after comma

def format_integer(x):
    return '{:,}'.format(int(x)).replace(',', ' ')

if __name__ == "__main__":
    N = 20
    n_range = [50, 100, 150, 200, 400, 600, 800, 1000]
    nb_experiments = 5
    K = 10000
    rho_factor = 2
    run_exact_carath = True
    filename=f"results/runtimes_nb_experiments_{nb_experiments}_N_{N}_nrange_{len(n_range)}_K_{K}_rho_{rho_factor}_bis.npy"
    runtimes = run_experiment(n_range=n_range, nb_experiments=nb_experiments, N=N, K=K, rho_factor=rho_factor, run_exact_carath=run_exact_carath)
    np.save(filename, runtimes)

        

        
        
        
