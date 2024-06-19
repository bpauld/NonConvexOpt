import os

# Set the environment variable
os.environ['SCIPY_USE_PROPACK'] = '1'

import numpy as np
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/bpauldub/ShapleyFolkman/')
sys.path.insert(1, '/home/bpauldub/ShapleyFolkman/code/')

from frank_wolfe_1 import *
from frank_wolfe_1_strongly_convex import *
from sparsify_constructive import *
from sparsify_frank_wolfe import frank_wolfe_2_fully_corrective, compute_f_values, build_final_solution_fcfw
from solve_dual_gd import *
import scipy as sp
import os
from utils import *
import csv




def run_experiment(n=50, N=10, nb_experiments=10):

    #rho_factor = 2
    #filename = f"results/results_rho_factor_{rho_factor}.csv"
    filename = f"results/results_final.csv"
    results = []
    for rs in range(nb_experiments):

        rho_factor = 0
        infeasibility_y_final_max = 1
        infeasibility_y_final_in_domain = 1
        infeasibility_y_final_fcfw_max = 1
        infeasibility_y_final_fcfw_in_domain = 1
        tol = 1e-6
        
        while max(infeasibility_y_final_max, infeasibility_y_final_in_domain, infeasibility_y_final_fcfw_max, infeasibility_y_final_fcfw_in_domain) > tol:
            rho_factor += 1
            uc_prob = create_uc_problem(n=n, N=N, random_seed=rs, rho_factor=rho_factor)
    
            A = uc_prob.construct_A_matrix()
    
            print(f"MaxMax generation = {np.max(uc_prob.max_gen)}, Sum(max_gen) - max(demand) = {np.sum(uc_prob.max_gen) - np.max(uc_prob.D)}") 
    
            v_star_rho = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=True)
            v_star = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=False)
    
            max_rho_fi = 0
            for i in range(uc_prob.n):
                rho_i = uc_prob.f_i(i, np.concatenate([np.ones(uc_prob.N), np.ones(uc_prob.N) * uc_prob.max_gen[i]]))
                max_rho_fi = rho_i if rho_i > max_rho_fi else max_rho_fi
    
            #======================= Run first conditional Gradient method =======================
            x_start = np.ones((uc_prob.di, uc_prob.n))
            for i in range(uc_prob.n):
                x_start[uc_prob.N:, i] = uc_prob.max_gen[i]
            K = 10000
            print(v_star_rho)
            z_K, grad_norm_list, y_dic, = frank_wolfe_1(v_star_rho, uc_prob.b_bar, K, uc_prob, x_start, verbose=True)
    
    
            #============ Sparsify using exact Caratheodory algorithm =========================
            Z_matrix, eta_vector, corresponding_indices = create_Z_matrix(y_dic, uc_prob)
            #eta_vector[eta_vector < 0] = 0
            print((Z_matrix @ eta_vector)[:uc_prob.m+1] - z_K)
            final_Z_matrix, final_eta_vector, final_corresponding_indices = sparsify_solution_caratheodory(Z_matrix, eta_vector, corresponding_indices, verbose=False)
    
    
            y_final, y_final_max, y_final_sampled = build_final_solution_caratheodory(y_dic, final_eta_vector, final_corresponding_indices)
            y_final_col = y_final.reshape((-1,), order='F')
            y_final_col_max = y_final_max.reshape((-1,), order='F')
            y_final_col_sampled = y_final_sampled.reshape((-1,), order='F')
            y_final_in_domain = uc_prob.build_solution_in_domain(y_final)
            y_final_col_in_domain = y_final_in_domain.reshape((-1,), order='F')
            print(A @ y_final_col_in_domain - uc_prob.b)
            print(A @ y_final_col_max - uc_prob.b)
            print(np.max(A @ y_final_col_in_domain - uc_prob.b))
            print(np.max(A @ y_final_col_max - uc_prob.b))
    
            infeasibility_y_final_max = np.clip(np.max((A @ y_final_col_max - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
            infeasibility_y_final_in_domain = np.clip(np.max((A @ y_final_col_in_domain - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
    
            f_y_final_max = uc_prob.f(y_final_max)
            f_y_final_domain = uc_prob.f(y_final_in_domain)
            print("Results Exact Carath : ", f_y_final_max, f_y_final_domain, infeasibility_y_final_max, infeasibility_y_final_in_domain)
    
    
            #============ Sparsify using approximate Caratheodory =================
            f_values = compute_f_values(y_dic, uc_prob)
            T = 3 * uc_prob.n # this should be enough
            nb_non_trivial_CC_wanted = uc_prob.N
            z_t, active_set, grad_list, opt_lambda, V_matrix, _ = frank_wolfe_2_fully_corrective(z_K, T, y_dic, uc_prob, f_values=f_values,
                                                                                                 nb_non_trivial_CC_wanted=nb_non_trivial_CC_wanted)
    
            y_final_fcfw, y_final_fcfw_max = build_final_solution_fcfw(y_dic, opt_lambda, active_set)
            y_final_fcfw_in_domain = uc_prob.build_solution_in_domain(y_final_fcfw)
            y_final_fcfw_col = y_final_fcfw.reshape((-1,), order='F')
            y_final_fcfw_col_max = y_final_fcfw_max.reshape((-1,), order='F')
            y_final_fcfw_col_in_domain = y_final_fcfw_in_domain.reshape((-1, ), order='F')
    
            infeasibility_y_final_fcfw_max = np.clip(np.max((A @ y_final_fcfw_col_max - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
            infeasibility_y_final_fcfw_in_domain = np.clip(np.max((A @ y_final_fcfw_col_in_domain - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
    
            f_y_final_fcfw_max = uc_prob.f(y_final_fcfw_max)
            f_y_final_fcfw_domain = uc_prob.f(y_final_fcfw_in_domain)
            print("Results approximate Carath : ", f_y_final_fcfw_max, f_y_final_fcfw_domain, infeasibility_y_final_fcfw_max, infeasibility_y_final_fcfw_in_domain)
        

        # ============== Save results ====================
        results.append([format_integer(v_star), format_integer(v_star_rho), format_integer(max_rho_fi), int(rho_factor), 
                        format_integer(f_y_final_max), format_integer(f_y_final_domain),
                       format_decimal(infeasibility_y_final_max), format_decimal(infeasibility_y_final_in_domain),
                        format_integer(f_y_final_fcfw_max), format_integer(f_y_final_fcfw_domain),
                       format_decimal(infeasibility_y_final_fcfw_max), format_decimal(infeasibility_y_final_fcfw_in_domain),
                       ])

    fields = ["v_star", "v_star_rho", "max_rho_fi", "rho_factor",
              "f_y_final_max", "f_y_final_domain",
             "infeasibility_y_final_max", "infeasibility_y_final_in_domain",
             "f_y_final_fcfw_max", "f_y_final_fcfw_domain",
             "infeasibility_y_fcfw_final_max", "infeasibility_y_final_fcfw_in_domain"]
    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(results)

def format_decimal(x):
    return '{:.1e}'.format(x)

def format_integer(x):
    return '{:,}'.format(int(x)).replace(',', ' ')

if __name__ == "__main__":
    run_experiment(nb_experiments=10)

        

        
        
        
