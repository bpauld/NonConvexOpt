import numpy as np
from convex_hull_optimization import optimize_over_convex_hull


def build_vector_from_indices(i, k, y_dic, prob, Ai=None):
    n = prob.n
    m = prob.m
    if Ai is None:
        Ai = prob.construct_Ai_matrix(i)
    v = np.zeros(1 + m + n)
    v[0] = prob.f_i(i, y_dic[i][0][:, k])
    v[1:m+1] = Ai @ y_dic[i][0][:, k]
    v[m+1 + i] = 1
    return v



def compute_f_values(y_dic, prob):
    n = prob.n
    f_values = {}
    for i in range(n):
        f_values[i] = np.zeros(y_dic[i][0].shape[1])
        for k in range(y_dic[i][0].shape[1]):
            f_values[i][k] = prob.f_i(i, y_dic[i][0][:, k])
    return f_values


def lmo_2(grad, y_dic, f_values, prob):
    n = prob.n
    m = prob.m
    best_index = (-1, -1)
    best_value = np.inf
    for i in range(n):
        Ai = prob.construct_Ai_matrix(i)
        Ai_y = np.dot(Ai, y_dic[i][0])
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


def frank_wolfe_2_fully_corrective(z_K, T, y_dic, prob, f_values=None, tol1=1e-2, tol2=1e-9, verbose=True,
                   solver_qp="proxqp", nb_non_trivial_CC_wanted="default"):
    n = prob.n
    m = prob.m

    if nb_non_trivial_CC_wanted == "default":
        nb_non_trivial_CC_wanted = m

    grad_list = []
    active_set = []
    is_index_added = np.zeros(n)
    V_matrix_list = []

    if f_values is None:
        f_values = compute_f_values(y_dic, prob)

    #compute z_star for convenience
    z_star = np.zeros(1 + m + n)
    z_star[:m+1] = z_K/n
    z_star[m+1:] = np.ones(n)/n


    z_t = np.zeros(n + m + 1)


    #pick index 0 for the first one
    first_i = np.random.randint(n)
    first_K = np.random.randint(y_dic[first_i][0].shape[1])
    print(first_i, first_K)
    z_t = build_vector_from_indices(first_i, first_K, y_dic, prob)
    active_set.append((first_i, first_K))
    is_index_added[first_i] += 1
    V_matrix_list = [z_t]
    nb_total_removals = 0

    
    for t in range(1, T):       
        grad_t = z_t - z_star

        grad_list.append(np.linalg.norm(grad_t))

        #now find the smallest among all the candidates
        best_i, best_k = lmo_2(grad_t, y_dic, f_values, prob)
        s_t = build_vector_from_indices(best_i, best_k, y_dic, prob)

        newly_added_i = best_i

        if (best_i, best_k) not in active_set:
            active_set.append((best_i, best_k))
            V_matrix_list.append(s_t)
            if verbose:
                print(f"Adding index ({best_i}, {best_k}) to active set")
            is_index_added[best_i] += 1
        else:
            if verbose:
                print(f"!!!!  Index {best_i}, {best_k} already in active set")

        #solve with proxqp
        z_t, opt_lambda = optimize_over_convex_hull(np.array(V_matrix_list).T, z_star, solver=solver_qp)
        
        min_opt_lambda = np.min(opt_lambda)
        if np.abs(np.sum(opt_lambda) - 1) > 1e-4 or np.min(opt_lambda) < -1e-4:
            if verbose:
                print(t, np.sum(opt_lambda), np.min(opt_lambda))

        nb_deleted_elements = 0
        matching_newly_added_i = np.array([(j, opt_lambda[j]) for j, tuple_ik in enumerate(active_set) if tuple_ik[0]==newly_added_i])
        if verbose:
            print(matching_newly_added_i[:, 1])
        argmin_matching_newly_added_i = int(matching_newly_added_i[np.argmin(matching_newly_added_i[:, 1]), 0])


        #remove extreme points which correspond to the same unit as the one we just added,
        #if their value in opt_lambda is smaller that tol1 multiplied by the value of opt_lambda at the newly added extreme point
        tol_tol = tol1 * np.max(matching_newly_added_i[:, 1])
        indices_to_remove = []
        for index in range(matching_newly_added_i.shape[0] - 1):
            #print(matching_newly_added_i[index])
            candidate_index = int(matching_newly_added_i[index][0])
            if opt_lambda[candidate_index] < tol_tol:
                indices_to_remove.append(candidate_index)        
                if verbose:
                    print(f"Removing index {active_set[candidate_index]} from active set")
                is_index_added[active_set[candidate_index][0]] -= 1
                nb_deleted_elements += 1
                nb_total_removals += 1

        # remove extreme points for which opt_lambda is too small, unless it is the last extreme point we added to the decomposition
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
        z_t, opt_lambda = optimize_over_convex_hull(np.array(V_matrix_list).T, z_star, solver=solver_qp)
        
        
        nb_missing_indices = np.sum(is_index_added == 0)
        nb_trivial_CC = np.sum(is_index_added == 1)
        nb_non_trivial_CC = np.sum(is_index_added > 1)
        if verbose:
            print(f"At iteration {t}, grad_norm = {np.linalg.norm(grad_t)}, min(opt_lambda) = {min_opt_lambda}, nb deleted = {nb_deleted_elements}, nb total deleted = {nb_total_removals}")
            print(f"In the final sum: missing indices = {nb_missing_indices}, trivial CC = {nb_trivial_CC}, non-trivial CC = {nb_non_trivial_CC}")

        
        V_matrix= np.array(V_matrix_list).T
        z_t = V_matrix @ opt_lambda

        if nb_missing_indices == 0 and nb_non_trivial_CC == nb_non_trivial_CC_wanted:
            break

    return z_t, np.array(active_set), grad_list, opt_lambda, np.array(V_matrix_list).T, is_index_added



def build_final_solution_fcfw(y_dic, opt_lambda, active_set):
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
