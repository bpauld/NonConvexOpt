import numpy as np
import sys
from convex_hull_optimization import optimize_over_convex_hull


def fully_corrective_frank_wolfe_1_strongly_convex(optimal_value_dual, Ax_star,
                 K, prob, x_start, tol=1e-2):

    n = prob.n
    m = prob.m
    di = prob.di
    
    z_star = np.zeros(1 + m)
    z_star[0] = optimal_value_dual
    z_star[1:] = Ax_star
    
    
    x_start_col = x_start.reshape((-1, ), order='F')
    
    z_k = np.zeros(1 + m)
    z_k[0] = prob.f(x_start)
    A = prob.construct_A_matrix()
    z_k[1:] = A.dot(x_start_col)


    pointer = 0
    y_dic = {}
    for i in range(n): 
        y_dic[i] = [np.zeros((di, 1)), [[0]]]
        y_dic[i][0][:, 0] = x_start[:, i]

    grad_norm_list = []
    g_k = 0
    active_set = [z_k]
    
    for k in range(1, K):
        grad_k = z_k - z_star
        grad_k_0 = grad_k[0]
        grad_k_rest = grad_k[1:]

        grad_norm = np.linalg.norm(grad_k)
        grad_norm_list.append(grad_norm)

        if k%1 == 0 or k==0:
        #if True:
            print(f"Iteration {k} : ||z_k - z*||_+^2 = {grad_k_0**2} + { np.linalg.norm(grad_k_rest)**2}")
            print(f"     ||grad(z_k)||/||z^*|| = {grad_norm/np.linalg.norm(z_star)}")
        if grad_norm/np.linalg.norm(z_star) < tol:
            break
        
        s_k = np.zeros(1 + m)

        #print(grad_k)
        for i in range(n):
            Ai = prob.construct_Ai_matrix(i)
            ATi_dot_gk = Ai.T.dot(grad_k_rest)
            y_ik = prob.lmo_1_linear(i, grad_k_0, ATi_dot_gk)

            f_biconjugate_ik = prob.f_i(i, y_ik)             

            Ati_dot_y_ik = Ai.dot(y_ik)
            s_k += np.concatenate(([f_biconjugate_ik], Ati_dot_y_ik))

            pointer_i = y_dic[i][1]
            #get index of where in the array where the current y_ik is in the dic
            # if it has not been seen before, this will return an empty array
            matching_y_ik = np.where(True == np.all(y_dic[i][0][:, :] == y_ik[:, None], axis=0))[0]
            if matching_y_ik.shape[0] == 0:
                y_dic[i][0] = np.concatenate((y_dic[i][0], y_ik.reshape(-1, 1)), axis=1)
                y_dic[i][1].append([k])
            else:
                y_dic[i][1][matching_y_ik[0]].append(k)

        active_set.append(s_k)       
        z_k, opt_lambda = optimize_over_convex_hull(np.array(active_set).T, z_star)              


    for i in range(n):
        for k in range(y_dic[i][0].shape[1]):
            #y_dic[i][1][k] = np.sum([opt_lambda[j] for j in y_dic[i][1][k]])
            y_dic[i][1][k] = np.clip(np.sum([opt_lambda[j] for j in y_dic[i][1][k]]), 0, None)
            
    return z_k, grad_norm_list, y_dic, opt_lambda



def is_in_active_set(s, active_set):
    
    equal_columns = np.all(np.array(active_set).T == s[:, np.newaxis], axis=0)
    # Find the index of the first column where equality holds
    column_number = np.argmax(equal_columns)
    # If no column is equal, return -1
    return (True, column_number) if equal_columns.any() else (False, -1)

def find_argmax_in_active_set(grad, active_set):
    index = np.argmax(np.array(active_set) @ grad)
    return active_set[index], index


def away_step_frank_wolfe_1_strongly_convex(optimal_value_dual, Ax_star,
                 K, prob, x_start, tol=1e-2):

    n = prob.n
    m = prob.m
    di = prob.di
    
    z_star = np.zeros(1 + m)
    z_star[0] = optimal_value_dual
    z_star[1:] = Ax_star
    
    
    x_start_col = x_start.reshape((-1, ), order='F')
    
    z_k = np.zeros(1 + m)
    z_k[0] = prob.f(x_start)
    A = prob.construct_A_matrix()
    z_k[1:] = A.dot(x_start_col)


    pointer = 0
    #y_dic = {}
    #for i in range(n): 
    #    y_dic[i] = [np.zeros((di, 1)), [[0]]]
    #    y_dic[i][0][:, 0] = x_start[:, i]

    grad_norm_list = []
    g_k = 0
    active_set = [z_k]
    active_set_alphas = np.ones(1)
    
    for k in range(1, K):
        grad_k = z_k - z_star
        grad_k_0 = grad_k[0]
        grad_k_rest = grad_k[1:]

        grad_norm = np.linalg.norm(grad_k)
        grad_norm_list.append(grad_norm)

        if k%100 == 0 or k==0:
        #if True:
            print(f"Iteration {k} : ||z_k - z*||_+^2 = {grad_k_0**2} + { np.linalg.norm(grad_k_rest)**2}")
            print(f"     ||grad(z_k)|| = {grad_norm}, ||grad(z_k)||/||z^*|| = {grad_norm/np.linalg.norm(z_star)}")
        if grad_norm/np.linalg.norm(z_star) < tol:
            break
        
        s_k = np.zeros(1 + m)

        #print(grad_k)
        for i in range(n):
            Ai = prob.construct_Ai_matrix(i)
            ATi_dot_gk = Ai.T.dot(grad_k_rest)
            y_ik = prob.lmo_1_linear(i, grad_k_0, ATi_dot_gk)

            f_biconjugate_ik = prob.f_i(i, y_ik)             

            Ati_dot_y_ik = Ai.dot(y_ik)
            s_k += np.concatenate(([f_biconjugate_ik], Ati_dot_y_ik))

            pointer_i = y_dic[i][1]
            #get index of where in the array where the current y_ik is in the dic
            # if it has not been seen before, this will return an empty array
            #matching_y_ik = np.where(True == np.all(y_dic[i][0][:, :] == y_ik[:, None], axis=0))[0]
            #if matching_y_ik.shape[0] == 0:
            #    y_dic[i][0] = np.concatenate((y_dic[i][0], y_ik.reshape(-1, 1)), axis=1)
            #    y_dic[i][1].append([k])
            #else:
            #    y_dic[i][1][matching_y_ik[0]].append(k)

        v_k, vk_index_in_active_set = find_argmax_in_active_set(grad_k, active_set)

        dk_FW = s_k - z_k
        dk_A = z_k - v_k

        FW_step = True
        if -grad_k.dot(dk_FW) >= -grad_k.dot(dk_A):
            dk = dk_FW
            gamma_max = 1
        else:
            FW_step = False
            dk = dk_A
            gamma_max = active_set_alphas[vk_index_in_active_set] / (1 - active_set_alphas[vk_index_in_active_set])

        gamma_k = min(gamma_max, -dk.dot(grad_k)/dk.dot(dk))
        z_k = z_k + gamma_k * dk

        if FW_step:
            if gamma_k == 1:
                active_set = [s_k]
                active_set_stepsize = np.ones(1)
            else:
                #active_set_alphas.append(0)
                sk_in_active_set, sk_index_in_active_set = is_in_active_set(s_k, active_set)
                if not sk_in_active_set:
                    active_set.append(s_k)
                    sk_index_in_active_set = len(active_set) - 1
                active_set_alphas = np.concatenate((active_set_alphas, [0])) 
                active_set_alphas = (1 - gamma_k) * active_set_alphas
                active_set_alphas[sk_index_in_active_set]+=  gamma_k
        else:
            if gamma_k == gamma_max:
                del active_set[vk_index_in_active_set]
                active_set_alphas = np.delete(active_set_alphas, vk_index_in_active_set)
            else:
                active_set_alphas = (1 + gamma_k) * active_set_alphas
                active_set_alphas[vk_index_in_active_set] -= gamma_k


        #print(gamma_k, gamma_max, FW_step, len(active_set), len(active_set_alphas))
                

            
    return z_k, grad_norm_list, y_dic