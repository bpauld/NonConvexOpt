import numpy as np
import time
from convex_hull_optimization import optimize_over_convex_hull

def clean_dictionary(y_dic):
    new_dic = {}
    for _, (i, value) in enumerate(y_dic.items()):
        pointer_i = value[1]
        new_array = value[0][:, :pointer_i]
        new_etas = value[2][:pointer_i]
        new_dic[i] = [new_array, new_etas]
    return new_dic  


def frank_wolfe_1(optimal_value_dual, b,
                 K, prob, x_start, keep_stepsizes=True, verbose=True):
    n = prob.n
    m = prob.m
    
    z_star = np.zeros(1 + m)
    z_star[0] = optimal_value_dual
    z_star[1:] = b
    
    
    x_start_col = x_start.reshape((-1, ), order='F')    
    z_k = np.zeros(1 + m)
    z_k[0] = prob.f(x_start)
    z_k[1:] = prob.compute_A_dot_x(x_start)

    pointer = 0
    y_dic = {}
    for i in range(n):
        y_dic[i] = np.zeros((prob.get_di(i), K))
        y_dic[i][:, 0] = x_start[:, i]
    etas_list = np.zeros(K)
    etas_list[0] = 1

    grad_norm_list = []
    g_k = 0

    
    total_time_lmo = 0
    start_time = time.time()
    
    for k in range(1, K):
        grad_k = np.clip(z_k - z_star, 0.0, None)
        grad_k_0 = grad_k[0]
        grad_k_rest = grad_k[1:]

        grad_norm = np.linalg.norm(grad_k)
        grad_norm_list.append(grad_norm)

        if verbose:
            frequency = int(K / 10)
            if k%frequency == 0 or k==0:
                print(f"Iteration {k} : ||z_k - z*||_+^2 = {grad_k_0**2} + { np.linalg.norm(grad_k_rest)**2}")
                print(f"     ||grad(z_k)|| = {grad_norm}, Time = {time.time() - start_time}, Total time LMO = {total_time_lmo}")
        
        start_time_lmo = time.time()
        y_k = prob.lmo_1(grad_k_0, grad_k_rest)

        f_biconjugate_yk = prob.f(y_k)             
        A_dot_yk = prob.compute_A_dot_x(y_k)
        s_k = np.concatenate(([f_biconjugate_yk], A_dot_yk))
        total_time_lmo += time.time() - start_time_lmo

        for i in range(n):
            y_ik = prob.get_y_ik(i, y_k)
            if keep_stepsizes:
                y_dic[i][:, k] = y_ik
            else:
                y_dic[i][:, k] = y_ik

        d_k = s_k - z_k
        g_k = -grad_k.dot(d_k)
        etak = min(1, g_k/d_k.dot(d_k))
        assert etak > 0
        
        #now update stepsize accounting
        if keep_stepsizes:
            etas_list *= (1-etak)
            etas_list[k] = etak
            
        
        
        z_k = (1 - etak) * z_k + etak * s_k               

    if keep_stepsizes:
        for i in range(n):
            u, unique_indices, unique_inverse = np.unique(y_dic[i], axis=1, return_index=True, return_inverse=True)
            etas = np.zeros(unique_indices.shape[0])
            #print(unique_indices, unique_inverse)
            for k in range(unique_inverse.shape[0]):
                loc = unique_inverse[k]
                etas[loc] += etas_list[k]
            #return
            y_dic[i] = [u, etas]
    else:
        for i in range(n):
            #keep y_dic[i] in the same order, this helps LMO of the next FW ?
            _, indices = np.unique(y_dic[i], axis=1, return_index=True)
            #print(indices.shape)
            y_dic[i] = [y_dic[i][:, np.sort(indices)], 0]
            
    return z_k, grad_norm_list, y_dic



def frank_wolfe_1_strongly_convex(optimal_value_dual, Ax_star,
                 K, prob, x_start, tol=1e-2, verbose=True, keep_stepsizes=True):

    n = prob.n
    m = prob.m
    
    z_star = np.zeros(1 + m)
    z_star[0] = optimal_value_dual
    z_star[1:] = Ax_star
    
    z_k = np.zeros(1 + m)
    z_k[0] = prob.f(x_start)
    z_k[1:] = prob.compute_A_dot_x(x_start)


    pointer = 0
    y_dic = {}
    for i in range(n):
        y_dic[i] = np.zeros((prob.get_di(i), K))
        y_dic[i][:, 0] = x_start[:, i]
    
    total_time_lmo = 0
    start_time = time.time()
    total_time_qp = 0

    grad_norm_list = []
    g_k = 0
    active_set = [z_k]
    
    for k in range(1, K):
        grad_k = z_k - z_star
        grad_k_0 = grad_k[0]
        grad_k_rest = grad_k[1:]

        grad_norm = np.linalg.norm(grad_k)
        grad_norm_list.append(grad_norm)

        if verbose:
            frequency = int(K / 10)
            if k%frequency == 0 or k==0:
                print(f"Iteration {k} : ||z_k - z*||_+^2 = {grad_k_0**2} + { np.linalg.norm(grad_k_rest)**2}")
                print(f"     Time = {time.time() - start_time}, Total time LMO = {total_time_lmo}, Total time QP = {total_time_qp}, size active set = {len(active_set)}")

        start_time_lmo = time.time()
        y_k = prob.lmo_1_linear(grad_k_0, grad_k_rest)

        f_biconjugate_yk = prob.f(y_k)             
        A_dot_yk = prob.compute_A_dot_x(y_k)
        s_k = np.concatenate(([f_biconjugate_yk], A_dot_yk))
        total_time_lmo += time.time() - start_time_lmo
        

        for i in range(n):
            y_ik = prob.get_y_ik(i, y_k)
            if keep_stepsizes:
                y_dic[i][:, k] = y_ik
            else:
                y_dic[i][:, k] = y_ik

        #if s_k not in active_set:
        active_set.append(s_k)       
        start_time_qp = time.time()
        z_k, etas_list = optimize_over_convex_hull(np.array(active_set).T, z_star)
        total_time_qp += time.time() - start_time_qp
        etas_list = np.where(etas_list < 0, 0, etas_list)


    if keep_stepsizes:
        for i in range(n):
            u, unique_indices, unique_inverse = np.unique(y_dic[i], axis=1, return_index=True, return_inverse=True)
            etas = np.zeros(unique_indices.shape[0])
            #print(unique_indices, unique_inverse)
            for k in range(unique_inverse.shape[0]):
                loc = unique_inverse[k]
                etas[loc] += etas_list[k]
            #return
            y_dic[i] = [u, etas]
    else:
        for i in range(n):
            #keep y_dic[i] in the same order, this helps LMO of the next FW ?
            _, indices = np.unique(y_dic[i], axis=1, return_index=True)
            #print(indices.shape)
            y_dic[i] = [y_dic[i][:, np.sort(indices)], 0]
            
    return z_k, grad_norm_list, y_dic
