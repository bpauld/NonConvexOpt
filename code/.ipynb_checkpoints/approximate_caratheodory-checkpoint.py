import numpy as np
from convex_hull_optimization import optimize_over_convex_hull
import scipy as sp
import time
from scipy.sparse import coo_matrix


def build_vector_from_indices(i, k, y_dic, prob, f_values=None):
    n = prob.n
    m = prob.m
    v = np.zeros(1 + m + n)
    if f_values is None:
        v[0] = prob.f_i(i, y_dic[i][0][:, k])
    else:
        v[0] = f_values[i][k]
    v[1:m+1] =  prob.compute_Ai_dot_y(i, y_dic[i][0][:, k])
    v[m+1 + i] = 1
    return v



def compute_f_values(y_dic, prob):
    n = prob.n
    f_values = {}
    for i in range(n):
        f_values[i] = prob.f_i_vec(i, y_dic[i][0])
        
    return f_values


def lmo_2(grad, y_dic, f_values, prob, Ai_y_dic):
    n = prob.n
    m = prob.m
    best_index = (-1, -1)
    best_value = np.inf
    for i in range(n):
        #Ai_y = prob.compute_Ai_dot_y(i, y_dic[i][0])
        Ai_y = Ai_y_dic[i]
        candidates = (
            f_values[i] * grad[0] +               # f_value(y_array[i, k]) * grad_t[0]
            np.dot(Ai_y.T, grad[1:m+1]) +              # A[:, i].dot(y_array[i, k]).dot(grad_t[1:m+1])
            grad[m+1+i]                              # grad_t[m+1+i]
        )

        min_candidate_index = np.argmin(candidates)
        min_candidate_value = candidates[min_candidate_index]
        if min_candidate_value < best_value:
            best_value = min_candidate_value
            best_index = (i, min_candidate_index)
    return best_index[0], best_index[1]


def sparsify_solution_approximate_caratheodory_fcfw(z_K, T, y_dic, prob, f_values=None, tol1=1e-2, tol2=1e-9, verbose=True,
                   solver_qp="proxqp", nb_non_trivial_CC_wanted="default", use_heuristic=False):
    n = prob.n
    m = prob.m

    start_time = time.time()

    if nb_non_trivial_CC_wanted == "default":
        nb_non_trivial_CC_wanted = m

    grad_list = []
    active_set = []
    is_index_added = np.zeros(n)
    V_matrix_list = []
    
    Ai_y_dic = {}
    for i in range(n):
        Ai_y_dic[i] = prob.compute_Ai_dot_y(i, y_dic[i][0]) 

    start_warmup = time.time()
    if f_values is None:
        f_values = compute_f_values(y_dic, prob)

    print(f"Warmup time = {time.time() - start_warmup}")

    #compute z_star for convenience
    z_star = np.zeros(1 + m + n)
    z_star[:m+1] = z_K/n
    z_star[m+1:] = np.ones(n)/n


    z_t = np.zeros(n + m + 1)


    #pick index 0 for the first one
    first_i = np.random.randint(n)
    first_K = np.random.randint(y_dic[first_i][0].shape[1])
    z_t = build_vector_from_indices(first_i, first_K, y_dic, prob)
    active_set.append((first_i, first_K))
    is_index_added[first_i] += 1
    V_matrix_list = [z_t]
    nb_total_removals = 0

    time_lmo = 0
    time_solve_qp = 0

    
    for t in range(1, T):       
        grad_t = z_t - z_star

        grad_list.append(np.linalg.norm(grad_t))

        #now find the smallest among all the candidates
        start_time_lmo = time.time()
        best_i, best_k = lmo_2(grad_t, y_dic, f_values, prob, Ai_y_dic)
        time_lmo += time.time() - start_time_lmo
        s_t = build_vector_from_indices(best_i, best_k, y_dic, prob)

        newly_added_i = best_i

        if (best_i, best_k) not in active_set:
            active_set.append((best_i, best_k))
            V_matrix_list.append(s_t)
            is_index_added[best_i] += 1
        else:
            if verbose:
                if nb_missing_indices == 0:
                    print(f"  Index {best_i}, {best_k} already in active set. Returning")
                    break

        #solve with proxqp
        start_time_solve_qp = time.time()
        z_t, opt_lambda = optimize_over_convex_hull(np.array(V_matrix_list).T, z_star, solver=solver_qp)
        time_solve_qp += time.time() - start_time_solve_qp
        
        min_opt_lambda = np.min(opt_lambda)
        if np.abs(np.sum(opt_lambda) - 1) > 1e-4 or np.min(opt_lambda) < -1e-4:
            if verbose:
                print(t, np.sum(opt_lambda), np.min(opt_lambda))

        nb_deleted_elements = 0
        matching_newly_added_i = np.array([(j, opt_lambda[j]) for j, tuple_ik in enumerate(active_set) if tuple_ik[0]==newly_added_i])
        argmin_matching_newly_added_i = int(matching_newly_added_i[np.argmin(matching_newly_added_i[:, 1]), 0])


        #remove extreme points which correspond to the same unit as the one we just added,
        #if their value in opt_lambda is smaller that tol1 multiplied by the value of opt_lambda at the newly added extreme point
        tol_tol = tol1 * np.max(matching_newly_added_i[:, 1])
        indices_to_remove = []
        if use_heuristic:
            for index in range(matching_newly_added_i.shape[0] - 1):
                candidate_index = int(matching_newly_added_i[index][0])
                if opt_lambda[candidate_index] < tol_tol:
                    indices_to_remove.append(candidate_index)        
                    if verbose:
                        print(f"Removing index {active_set[candidate_index]} from active set")
                    is_index_added[active_set[candidate_index][0]] -= 1
                    nb_deleted_elements += 1
                    nb_total_removals += 1

        # remove extreme points for which opt_lambda is too small, unless it is the last extreme point we added to the decomposition
        if use_heuristic:
            for candidate_index in range(len(opt_lambda) - 1):
                if opt_lambda[candidate_index] < tol2 and candidate_index not in indices_to_remove:
                    if verbose:
                        print(f"Removing index {active_set[candidate_index]} from active set")
                    is_index_added[active_set[candidate_index][0]] -= 1
                    indices_to_remove.append(candidate_index)
                    nb_deleted_elements += 1
                    nb_total_removals += 1
                
                
            new_active_set = [active_set[i] for i in range(len(active_set)) if i not in indices_to_remove]
            new_V_matrix_list = [V_matrix_list[i] for i in range(len(V_matrix_list)) if i not in indices_to_remove]
            active_set = new_active_set
            V_matrix_list = new_V_matrix_list
            opt_lambda = np.delete(opt_lambda, indices_to_remove)   
    
            #reoptimize ?
            #z_t, opt_lambda = optimize_over_convex_hull(np.array(V_matrix_list).T, z_star, solver=solver_qp)
        
        #QP solver might give slightly negative values -> set them to 0
        opt_lambda = np.where(opt_lambda < 0, 0, opt_lambda)
        
        
        nb_missing_indices = np.sum(is_index_added == 0)
        nb_trivial_CC = np.sum(is_index_added == 1)
        nb_non_trivial_CC = np.sum(is_index_added > 1)
        if verbose:
            frequency = int(prob.n / 10)
            if t % frequency == 0:
                total_time = time.time() - start_time
                print(f"At iteration {t}, grad_norm = {np.linalg.norm(grad_t)}, min(opt_lambda) = {min_opt_lambda}")
                print(f"    Total time = {total_time},  time LMO = {time_lmo}, time solve QP = {time_solve_qp}")
                print(f"    In the final sum: missing indices = {nb_missing_indices}, trivial CC = {nb_trivial_CC}, non-trivial CC = {nb_non_trivial_CC}")

        
        V_matrix= np.array(V_matrix_list).T
        z_t = V_matrix @ opt_lambda

        if nb_missing_indices == 0 and nb_non_trivial_CC >= nb_non_trivial_CC_wanted:
            print(f"    Finishing after {t} iterations with missing indices = {nb_missing_indices}, trivial CC = {nb_trivial_CC}, non-trivial CC = {nb_non_trivial_CC}")
            break
    return z_t, np.array(active_set), grad_list, opt_lambda, np.array(V_matrix_list).T, is_index_added


def create_AK(y_dic, prob):
    nb_columns = 0
    for i in range(prob.n):
        nb_columns += y_dic[i][0].shape[1]
    AK_dense_part = np.zeros((1 + prob.m, nb_columns))
    AK_last_row = []
    curr_index = 0
    for i in range(prob.n):
        for k in range(y_dic[i][0].shape[1]):
            AK_dense_part[0, curr_index] = prob.f_i(i, y_dic[i][0][:, k])
            AK_dense_part[1:prob.m+1, curr_index] = prob.compute_Ai_dot_y(i, y_dic[i][0][:, k])
            AK_last_row.append(i)
            curr_index += 1
    return AK_dense_part, AK_last_row


def lmo2_AK(z, AK_dense_part, AK_last_row, prob):
    index = np.argmin(AK_dense_part.T @ z[:prob.m+1] + z[prob.m+1:][AK_last_row])
    i = AK_last_row[index]
    k = np.sum([1 for j in range(index) if AK_last_row[j]==i])
    return i, int(k)

def build_small_vector_from_indices(i, k, y_dic, prob, f_values=None):
    m = prob.m
    v = np.zeros(1 + m)
    if f_values is None:
        v[0] = prob.f_i(i, y_dic[i][0][:, k])
    else:
        v[0] = f_values[i][k]
    v[1:m+1] =  prob.compute_Ai_dot_y(i, y_dic[i][0][:, k])
    return v


def initialize_PS(i, k, z_star, y_dic, prob):
    P_S = np.zeros((prob.m + 1, 1))
    P_S[:, 0] = build_small_vector_from_indices(i, k, y_dic, prob) - z_star[0:prob.m+1]
    P_S_index_list = [i]
    return P_S, P_S_index_list

def update_PS(P_S, P_S_index_list, i, k, z_star, y_dic, prob):
    P_i_k = build_small_vector_from_indices(i, k, y_dic, prob) - z_star[0:prob.m+1]
    P_S = np.c_[P_S, P_i_k]
    P_S_index_list.append(i)
    return P_S, P_S_index_list
    

def compute_PS_dot_v(v, P_S, P_S_index_list, z_star, prob):
    res = np.zeros(prob.n + prob.m + 1)
    res[0:prob.m+1] = P_S @ v
    for (index_in_list, i) in enumerate(P_S_index_list):
        res[prob.m +1 + i] += v[index_in_list]
    res[prob.m+1:] -= z_star[prob.m+1:]
    return res

def compute_PS_T_dot_v(v, P_S, P_S_index_list, z_star, prob):
    #res = np.zeros(P_S.shape[1])
    res = P_S.T @ v[:prob.m+1]
    res += v[prob.m+1:][P_S_index_list]
    res = res - z_star[prob.m+1:].dot(v[prob.m+1:])
    return res


# Implementation of Wolfe's min point norm algorithm from paper
# "FINDING THE NEAREST POINT IN A POLYTOPE" 
# using the sparsity of the vectors in A_K to never form the matrix P[S] 
# and using the implementation trick D from the paper to only ever solve triangular linear systems.
def sparsify_solution_approximate_caratheodory_mnp(z_K, T, y_dic, prob, lmo_method="method1", f_values=None, Z1=1e-12, Z2=1e-10, Z3=1e-10, verbose=True):
    n = prob.n
    m = prob.m

    grad_list = []
    active_set = []
    is_index_added = np.zeros(n)
    
    start_time = time.time()

    if lmo_method=="method2":
        print("Creating matrix AK")
        AK_dense_part, AK_last_row = create_AK(y_dic, prob)
        print(f"Matrix AK created with size {AK_dense_part.shape}")

    #compute z_star for convenience
    z_star = np.zeros(1 + m + n)
    z_star[:m+1] = z_K/n
    z_star[m+1:] = np.ones(n)/n

    
    #build Ai_y_dic
    Ai_y_dic = {}
    for i in range(n):
        Ai_y_dic[i] = prob.compute_Ai_dot_y(i, y_dic[i][0]) 

    start_warmup = time.time()
    if f_values is None:
        f_values = compute_f_values(y_dic, prob)

    print(f"Warmup time = {time.time() - start_warmup}")



    time_R = time.time()
    best_index = (-1, -1)
    best_value = np.inf
    for i in range(n):
        Ai_y = Ai_y_dic[i]
        P_i = np.zeros((1 + m, f_values[i].shape[0]))
        P_i[0, :] = f_values[i]
        P_i[1:m+1, :] = Ai_y
        P_i -= z_star[:m+1, None]
        candidates = np.sum(P_i**2, axis=0)
        #print(candidates_i.shape)

        min_candidate_index = np.argmin(candidates)
        min_candidate_value = candidates[min_candidate_index]
        if min_candidate_value < best_value:
            best_value = min_candidate_value
            best_index = (i, min_candidate_index)
    best_i, best_k = best_index

    print(f"Time to compute initial P_j new = {time.time() - time_R}")


    active_set = [(best_i, best_k)]
    opt_lambda = np.ones(1)
    #P_S = np.zeros((n+m+1, 1))
    #P_S[:, 0] = build_vector_from_indices(best_i, best_k, y_dic, prob) - z_star
    P_S_small, P_S_index_list = initialize_PS(best_i, best_k, z_star, y_dic, prob)
    
    #R = np.sqrt(1 + np.linalg.norm(P_S[:, 0])**2) * np.ones((1, 1)) 
    #preallocate the max size of R
    P_S_0 = build_vector_from_indices(best_i, best_k, y_dic, prob) - z_star
    R_big = np.zeros((n+m+1, n+m+1)) #prob never bigger than this ?
    pointer_R_big = 1
    R_big[0, 0] = np.sqrt(1 + np.linalg.norm(P_S_0)**2)
    
    time_lmo = 0
    time_linear_system = 0

    norms_P_S = [P_S_0.dot(P_S_0)]

    for t in range(1, T):
        # Step 1 of MNP
        #implement step 1(b) and obtain (Ji, Jk)
        z_t = compute_PS_dot_v(opt_lambda, P_S_small, P_S_index_list, z_star, prob)

        max_norm_P_S = max(norms_P_S)
        
        start_time_lmo = time.time()
        if lmo_method=="method2":
            Ji, Jk = lmo2_AK(z_t, AK_dense_part, AK_last_row, prob)
        else:
            #Ji, Jk = lmo_2(z_t, y_dic, f_values, prob)
            Ji, Jk = lmo_2(z_t, y_dic, f_values, prob, Ai_y_dic)
        time_lmo = time_lmo + time.time() - start_time_lmo
        P_Ji_Jk = build_vector_from_indices(Ji, Jk, y_dic, prob) - z_star
        
        #find max |P_J| for J \in active set
        if z_t.dot(P_Ji_Jk) > z_t.dot(z_t) - Z1 * max(P_Ji_Jk.dot(P_Ji_Jk), max_norm_P_S):
            print(f"Break at 1(c) with size active set = {len(active_set)}")
            break
        elif (Ji, Jk) in active_set:
            print("Break at 1(d)")
            break
        else:
            
            active_set.append((Ji, Jk))
            opt_lambda = np.concatenate((opt_lambda, [0])) 
            #update R
            start_time_ls = time.time()
            r = sp.linalg.solve_triangular(R_big[:pointer_R_big, :pointer_R_big].T, 1 + compute_PS_T_dot_v(P_Ji_Jk, P_S_small, P_S_index_list, z_star, prob), lower=True)
            #r = sp.linalg.solve_triangular(R.T, 1 + P_S.T @ P_Ji_Jk, lower=True)
            time_linear_system += time.time() - start_time_ls

            
            rho = np.sqrt(1 + P_Ji_Jk.dot(P_Ji_Jk) - r.dot(r))
            new_col = np.concatenate((r, [rho]))

            #update R_big
            if pointer_R_big == R_big.shape[1]:
                #this should only happen if we have deleted many columns of R_big
                print("Adding columns to R_big")
                nb_new_cols = 100
                R_big_new = np.zeros((n+m+1, R_big.shape[1] + nb_new_cols))
                R_big_new[:, :R_big.shape[1]] = R_big
                R_big = R_big_new
            R_big[:pointer_R_big+1, pointer_R_big] = new_col
            pointer_R_big += 1
            #update P_S
            P_S_small, P_S_index_list = update_PS(P_S_small, P_S_index_list, Ji, Jk, z_star, y_dic, prob)
            norms_P_S.append(P_Ji_Jk.dot(P_Ji_Jk))

        v = - np.zeros(1)
        counter = 0
        while np.any(v < Z2):
            counter += 1
            # Step 2 of MNP
            start_time_ls = time.time()
            u_bar = sp.linalg.solve_triangular(R_big[:pointer_R_big, :pointer_R_big].T, np.ones(pointer_R_big), lower=True)
            u = sp.linalg.solve_triangular(R_big[:pointer_R_big, :pointer_R_big], u_bar, lower=False)
            time_linear_system += time.time() - start_time_ls
            v = u / np.sum(u)
            #Step 2(b)
            if np.all(v > Z2):
                opt_lambda = v
                break
            else:
                pos = np.where(opt_lambda - v > Z3, opt_lambda/(opt_lambda - v), np.inf)
                theta = min(1, np.min(pos))
                #opt_lambda = theta * opt_lambda + (1-theta)*v
                opt_lambda = ( 1- theta) * opt_lambda + theta*v
                opt_lambda = np.where( opt_lambda < Z2, 0, opt_lambda)
                indices_to_remove = np.where(opt_lambda==0)
                if indices_to_remove[0].shape[0] > 1:
                    print("more than one index to remove, this might cause problems")
                I = indices_to_remove[0][0]
                
                opt_lambda = np.delete(opt_lambda, I)
                P_S_small = np.delete(P_S_small, I, axis=1)
                del P_S_index_list[I]
                del norms_P_S[I]
                del active_set[I]

                #update R
                R_big = np.delete(R_big, I, axis=1)
                pointer_R_big -= 1
                while I < pointer_R_big:
                    a = R_big[I, I]
                    b = R_big[I+1, I]
                    c = np.sqrt(a**2 + b**2)
                    new_R_I = (a * R_big[I, :] + b * R_big[I+1, :]) / c
                    new_R_I1 = (-b * R_big[I, :] + a * R_big[I+1, :])/c
                    R_big[I, :] = new_R_I
                    R_big[I+1, :] = new_R_I1
                    I += 1
                #at this point the last row of R should be 0
                    #print(R_big[pointer_R_big, :])
                assert np.linalg.norm(R_big[pointer_R_big, :]) < 1e-6
                

        if verbose:
            freq = int(prob.n / 10)
            if t % freq == 0:
                total_time = time.time() - start_time
                print(f"At iteration {t}, ||z_t||^2 = {np.linalg.norm(z_t)**2}, size active set = {len(active_set)}")
                print(f"    Total time = {total_time}, time LMO = {time_lmo}, time solving LS = {time_linear_system}")
    return z_t, active_set, opt_lambda




def find_argmax_in_active_set(grad, active_set, y_dic, prob):
    best_index = -1
    best_value = - np.inf
    best_vec = 0
    for (index, (i, k)) in enumerate(active_set):
        v_i_k = build_vector_from_indices(i, k, y_dic, prob)
        if grad.dot(v_i_k) > best_value:
            best_value = grad.dot(v_i_k)
            best_index = index
            best_vec = v_i_k
    return best_vec, best_index

def build_final_solution_approximate_caratheodory(y_dic, opt_lambda, active_set):
    n = len(y_dic)
    di = y_dic[0][0].shape[0]
    y_final = np.zeros((di, n))
    y_final_max = np.zeros((di, n))
    
    for i in range(n):
        matching_indices = [(index, j, k) for index, (j, k) in enumerate(active_set) if i == j]
        sum_lambda = sum([opt_lambda[int(index)] for (index, j, k) in matching_indices])
        max_value = -1

        for (index, j, k) in matching_indices:
            y_final[:, i] += 1/sum_lambda * opt_lambda[index] * y_dic[int(j)][0][:, int(k)] 

            if opt_lambda[index] > max_value:
                max_value = opt_lambda[index]
                y_final_max[:, i] = y_dic[int(j)][0][:, int(k)]
     
    return y_final, y_final_max
