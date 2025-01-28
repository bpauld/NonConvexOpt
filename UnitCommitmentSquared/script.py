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
from approximate_caratheodory import sparsify_solution_approximate_caratheodory_fcfw, compute_f_values, build_final_solution_approximate_caratheodory, sparsify_solution_approximate_caratheodory_mnp
from solve_dual_gd import *
import scipy as sp
import os
from utils import *
import csv
import time




def run_experiment(n=50, K=10000, N=10, nb_experiments=10, filename=f"results/results_final.csv"):

    #rho_factor = 2
    #filename = f"results/results_rho_factor_{rho_factor}.csv"
    results = []
    for rs in range(nb_experiments):

        rho_factor_exact_carath = 0
        rho_factor_approx_carath_fcfw = 0
        rho_factor_approx_carath_mnp = 0
        infeasibility_y_final_max = 1
        infeasibility_y_final_in_domain = 1
        infeasibility_y_final_fcfw_max = 1
        infeasibility_y_final_fcfw_in_domain = 1
        infeasibility_y_final_mnp_max = 1
        infeasibility_y_final_mnp_in_domain = 1
        tol = 1e-6
        runtime_exact_carath = 0
        runtime_approx_carath_fcfw = 0
        runtime_approx_carath_mnp = 0
        
        
        #====================================================================================================================================================
        #first run the whole procedure with exact Caratheodory
        while infeasibility_y_final_in_domain > tol:
            rho_factor_exact_carath += 1
            print(f"RUNNING EXACT CARATH WITH RHO = {rho_factor_exact_carath}")
            uc_prob = create_uc_problem(n=n, N=N, random_seed=rs, rho_factor=rho_factor_exact_carath)
    
            #A = uc_prob.construct_A_matrix()
    
            print(f"MaxMax generation = {np.max(uc_prob.max_gen)}, Sum(max_gen) - max(demand) = {np.sum(uc_prob.max_gen) - np.max(uc_prob.D)}") 
        
            start_time_first_stage = time.time()
    
            v_star_rho = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=True)
            v_star = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=False)
    
            max_rho_fi = 0
            for i in range(uc_prob.n):
                rho_i = uc_prob.f_i(i, np.concatenate([np.ones(uc_prob.N), np.ones(uc_prob.N) * uc_prob.max_gen[i]]))
                max_rho_fi = rho_i if rho_i > max_rho_fi else max_rho_fi
    
            #======================= Run first conditional Gradient method =======================
            x_start = uc_prob.get_feasible_point() # or initiate other feasible point
            print(f"Optimal dual value of perturbed problem = {v_star_rho}")
            z_K, grad_norm_list, y_dic, = frank_wolfe_1(v_star_rho, uc_prob.b_bar, K, uc_prob, x_start, verbose=True, keep_stepsizes=True)
            runtime_first_stage = time.time() - start_time_first_stage
    
    
            #============ Sparsify using exact Caratheodory algorithm =========================
            
            start_time_exact_carath = time.time()
            Z_matrix, eta_vector, corresponding_indices = create_Z_matrix(y_dic, uc_prob)
            #eta_vector[eta_vector < 0] = 0
            final_Z_matrix, final_eta_vector, final_corresponding_indices = sparsify_solution_exact_caratheodory(Z_matrix, eta_vector, corresponding_indices, verbose=False)
            runtime_exact_carath = time.time() - start_time_exact_carath
            total_time_exact_carath = runtime_first_stage + runtime_exact_carath





            y_final, y_final_max, y_final_sampled = build_final_solution_exact_caratheodory(y_dic, final_eta_vector, final_corresponding_indices)
            y_final_col = y_final.reshape((-1,), order='F')
            y_final_col_max = y_final_max.reshape((-1,), order='F')
            y_final_col_sampled = y_final_sampled.reshape((-1,), order='F')
            y_final_in_domain = uc_prob.build_solution_in_domain(y_final)
            y_final_col_in_domain = y_final_in_domain.reshape((-1,), order='F')

            infeasibility_y_final_max = np.clip(np.max((A @ y_final_col_max - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
            infeasibility_y_final_in_domain = np.clip(np.max((A @ y_final_col_in_domain - uc_prob.b) / np.abs(uc_prob.b)), 0, None)

            f_y_final_max = uc_prob.f(y_final_max)
            f_y_final_domain = uc_prob.f(y_final_in_domain)
            print("Results Exact Carath : ", f_y_final_max, f_y_final_domain, infeasibility_y_final_max, infeasibility_y_final_in_domain, runtime_exact_carath)

        print("")
        print("DONE WITH EXACT CARATHEODORY")
        print("")
        #============================================================================================================================
        # Now run the whole procedure with approximate caratheodory FCFW
        while infeasibility_y_final_fcfw_in_domain > tol:
            rho_factor_approx_carath_fcfw += 1
            print(f"RUNNING FCFW WITH RHO = {rho_factor_approx_carath_fcfw}")
            uc_prob = create_uc_problem(n=n, N=N, random_seed=rs, rho_factor=rho_factor_approx_carath_fcfw)
    
            A = uc_prob.construct_A_matrix()
    
            print(f"MaxMax generation = {np.max(uc_prob.max_gen)}, Sum(max_gen) - max(demand) = {np.sum(uc_prob.max_gen) - np.max(uc_prob.D)}") 
        
            start_time_first_stage = time.time()
    
            v_star_rho = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=True)
            v_star = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=False)
    
            max_rho_fi = 0
            for i in range(uc_prob.n):
                rho_i = uc_prob.f_i(i, np.concatenate([np.ones(uc_prob.N), np.ones(uc_prob.N) * uc_prob.max_gen[i]]))
                max_rho_fi = rho_i if rho_i > max_rho_fi else max_rho_fi
    
            #======================= Run first conditional Gradient method =======================
            x_start = uc_prob.get_feasible_point() # or initiate other feasible point
            print(f"f(x_start) = {uc_prob.f(x_start)}")
            keep_stepsizes=False
            z_K, grad_norm_list, y_dic, = frank_wolfe_1(v_star_rho, uc_prob.b_bar, K, uc_prob, x_start, verbose=True, keep_stepsizes=keep_stepsizes)
            
            runtime_first_stage = time.time() - start_time_first_stage
    
    
            #============ Sparsify using approximate Caratheodory FCFW =================
            start_time_approx_carath = time.time()
            f_values = compute_f_values(y_dic, uc_prob)
            T = 3 * uc_prob.n # this should be enough
            nb_non_trivial_CC_wanted = uc_prob.N 
            print(f"Running FCFW for {T} iterations")
            z_t, active_set, grad_list, opt_lambda, V_matrix, _ = sparsify_solution_approximate_caratheodory_fcfw(z_K, T, y_dic, uc_prob, f_values=f_values,                                                                                           nb_non_trivial_CC_wanted=nb_non_trivial_CC_wanted)
            
            runtime_approx_carath_fcfw = time.time() - start_time_approx_carath
            total_time_approx_carath_fcfw = runtime_first_stage + runtime_approx_carath_fcfw
            
    
            y_final_fcfw, y_final_fcfw_max = build_final_solution_approximate_caratheodory(y_dic, opt_lambda, active_set)
            y_final_fcfw_in_domain = uc_prob.build_solution_in_domain(y_final_fcfw)
            y_final_fcfw_col = y_final_fcfw.reshape((-1,), order='F')
            y_final_fcfw_col_max = y_final_fcfw_max.reshape((-1,), order='F')
            y_final_fcfw_col_in_domain = y_final_fcfw_in_domain.reshape((-1, ), order='F')
    
            infeasibility_y_final_fcfw_max = np.clip(np.max((A @ y_final_fcfw_col_max - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
            infeasibility_y_final_fcfw_in_domain = np.clip(np.max((A @ y_final_fcfw_col_in_domain - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
    
            f_y_final_fcfw_max = uc_prob.f(y_final_fcfw_max)
            f_y_final_fcfw_domain = uc_prob.f(y_final_fcfw_in_domain)
            print(f"Results approximate Carath FCFW: rho = {rho_factor_approx_carath_fcfw}, f(y_max) = {f_y_final_fcfw_max}, f(y_domain) = {f_y_final_fcfw_domain}, infeas_y_max = {infeasibility_y_final_fcfw_max}, infeas_y_domain = {infeasibility_y_final_fcfw_in_domain}, runtime approximate carath = {runtime_approx_carath_fcfw}, total time = {total_time_approx_carath_fcfw}")

        #===================================================================================================================
        # Now run the whole procedure with approximate caratheodory MNP
        while infeasibility_y_final_mnp_in_domain > tol:
            rho_factor_approx_carath_mnp += 1
            print(f"RUNNING MNP WITH RHO = {rho_factor_approx_carath_mnp}")
        
            uc_prob = create_uc_problem(n=n, N=N, random_seed=rs, rho_factor=rho_factor_approx_carath_mnp)
    
            A = uc_prob.construct_A_matrix()
    
            print(f"MaxMax generation = {np.max(uc_prob.max_gen)}, Sum(max_gen) - max(demand) = {np.sum(uc_prob.max_gen) - np.max(uc_prob.D)}") 
        
            start_time_first_stage = time.time()
    
            v_star_rho = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=True)
            v_star = solve_dual_gd(uc_prob, eta="1/k", max_iter=1000, solve_contracted_problem=False)
    
            max_rho_fi = 0
            for i in range(uc_prob.n):
                rho_i = uc_prob.f_i(i, np.concatenate([np.ones(uc_prob.N), np.ones(uc_prob.N) * uc_prob.max_gen[i]]))
                max_rho_fi = rho_i if rho_i > max_rho_fi else max_rho_fi
    
            #======================= Run first conditional Gradient method =======================
            x_start = uc_prob.get_feasible_point() # or initiate other feasible point
            print(f"f(x_start) = {uc_prob.f(x_start)}")
            keep_stepsizes=False
            z_K, grad_norm_list, y_dic, = frank_wolfe_1(v_star_rho, uc_prob.b_bar, K, uc_prob, x_start, verbose=True, keep_stepsizes=keep_stepsizes)
            
            runtime_first_stage = time.time() - start_time_first_stage
    
    
            #============ Sparsify using approximate Caratheodory MNP =================
            start_time_approx_carath = time.time()
            f_values = compute_f_values(y_dic, uc_prob)
            T = 10 * uc_prob.n # this should be enough

            z_t, active_set, opt_lambda = sparsify_solution_approximate_caratheodory_mnp(z_K, T, y_dic, uc_prob, f_values=f_values)
            
            runtime_approx_carath_mnp = time.time() - start_time_approx_carath
            total_time_approx_carath_mnp = runtime_first_stage + runtime_approx_carath_mnp
            
    
            y_final_mnp, y_final_mnp_max = build_final_solution_approximate_caratheodory(y_dic, opt_lambda, active_set)
            y_final_mnp_in_domain = uc_prob.build_solution_in_domain(y_final_mnp)
            y_final_mnp_col = y_final_mnp.reshape((-1,), order='F')
            y_final_mnp_col_max = y_final_mnp_max.reshape((-1,), order='F')
            y_final_mnp_col_in_domain = y_final_mnp_in_domain.reshape((-1, ), order='F')
    
            infeasibility_y_final_mnp_max = np.clip(np.max((A @ y_final_mnp_col_max - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
            infeasibility_y_final_mnp_in_domain = np.clip(np.max((A @ y_final_mnp_col_in_domain - uc_prob.b) / np.abs(uc_prob.b)), 0, None)
    
            f_y_final_mnp_max = uc_prob.f(y_final_mnp_max)
            f_y_final_mnp_domain = uc_prob.f(y_final_mnp_in_domain)
            print(f"Results approximate Carath MNP: rho = {rho_factor_approx_carath_mnp}, f(y_max) = {f_y_final_mnp_max}, f(y_domain) = {f_y_final_mnp_domain}, infeas_y_max = {infeasibility_y_final_mnp_max}, infeas_y_domain = {infeasibility_y_final_mnp_in_domain}, runtime approximate carath = {runtime_approx_carath_mnp}, total time = {total_time_approx_carath_mnp}")



        
        

        # ============== Save results ====================
        results.append([format_integer(v_star), format_integer(max_rho_fi), int(rho_factor_exact_carath), int(rho_factor_approx_carath_fcfw), int(rho_factor_approx_carath_mnp), 
                        format_integer(f_y_final_max), format_integer(f_y_final_domain),
                       format_decimal(infeasibility_y_final_max), format_decimal(infeasibility_y_final_in_domain),
                        format_integer(f_y_final_fcfw_max), format_integer(f_y_final_fcfw_domain),
                       format_decimal(infeasibility_y_final_fcfw_max), format_decimal(infeasibility_y_final_fcfw_in_domain),
                        format_integer(f_y_final_mnp_max), format_integer(f_y_final_mnp_domain),
                       format_decimal(infeasibility_y_final_mnp_max), format_decimal(infeasibility_y_final_mnp_in_domain),
                        format_time(runtime_first_stage), format_time(runtime_exact_carath), format_time(runtime_approx_carath_fcfw),
                        format_time(runtime_approx_carath_mnp)
                       ])

    fields = ["v_star", "max_rho_fi", "rho_factor_exact_carath", "rho_factor_approx_carath_fcfw", "rho_factor_approx_carath_mnp",
              "f_y_final_max", "f_y_final_domain",
             "infeasibility_y_final_max", "infeasibility_y_final_in_domain",
             "f_y_final_fcfw_max", "f_y_final_fcfw_domain",
             "infeasibility_y_fcfw_final_max", "infeasibility_y_final_fcfw_in_domain",
              "f_y_final_mnp_max", "f_y_final_mnp_domain",
             "infeasibility_y_mnp_final_max", "infeasibility_y_final_mnp_in_domain",
              "runtime_first_stage", "runtime_exact_carath", "runtime_approx_carath_fcfw", "runtime_approx_carath_mnp"]
    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(results)

def format_decimal(x):
    return '{:.1e}'.format(x)

def format_time(x):
    return str(round(x, 2)) #only keep 2 digits after comma

def format_integer(x):
    return '{:,}'.format(int(x)).replace(',', ' ')

if __name__ == "__main__":
    n = 50
    N = 10
    run_experiment(n=n, K=10000, N=N, nb_experiments=10, filename=f"results/results_n_{n}_N_{N}.csv")

        

        
        
        
