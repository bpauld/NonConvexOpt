import numpy as np
import scipy as sp


def create_Z_matrix(y_dic, prob):
    n = prob.n
    m = prob.m
    nb_of_total_elements = 0
    for i in range(n):
        nb_of_total_elements += y_dic[i][0].shape[1]
    counter = 0
    Z_matrix = np.zeros((n + m + 1, nb_of_total_elements))
    eta_vector = np.zeros(nb_of_total_elements)
    corresponding_indices = []
    
    for i in range(n):
        for k in range(y_dic[i][0].shape[1]):
            y_ik = y_dic[i][0][:, k]
            Z_matrix[0, counter] = prob.f_i(i, y_ik)
            Z_matrix[1:m+1, counter] = prob.compute_Ai_dot_y(i, y_ik)
            Z_matrix[m + 1 + i, counter] = 1
            eta_vector[counter] = y_dic[i][1][k]
            corresponding_indices.append((i, k))
            counter += 1
    
    return Z_matrix, eta_vector, corresponding_indices


def find_element_in_nullspace(Z, method='fastest'):
    if method=='full_nullspace':
        nullspace = sp.linalg.null_space(Z) # there is probably a more efficient way to do this
        if nullspace.shape[1] == 0:
            return 0, False
        else:
            return nullspace[:, 0], True 
    elif method == 'qr':
        if Z.shape[1] <= Z.shape[0]:
            return 0, False
        else:
            Q = np.linalg.qr(Z.T).Q
            random_line = np.random.randn(Q.shape[0])
            new_Q = np.zeros((Q.T.shape[0] + 1, Q.T.shape[1]))
            new_Q[:-1, :] = Q.T
            new_Q[-1, :] = random_line
            rhs = np.zeros(Z.shape[0] + 1)
            rhs[-1] = 100
            sol = np.linalg.lstsq(new_Q, rhs)
            return sol[0], True
    elif method == 'fastest':
        if Z.shape[1] <= Z.shape[0]:
            return 0, False
        else:
            random_line = np.random.randn(Z.shape[1])
            new_Z = np.zeros((Z.shape[0] + 1, Z.shape[1]))
            new_Z[:-1, :] = Z
            new_Z[-1, :] = random_line
            rhs = np.zeros(Z.shape[0] + 1)
            rhs[-1] = 100
            sol = np.linalg.lstsq(new_Z, rhs)
            return sol[0], True      



def sparsify_solution_exact_caratheodory(Z_matrix, eta_vector, corresponding_indices, max_iter=int(1e7), verbose=True):

    eta_vector_copy = eta_vector.copy()
    final_corresponding_indices = corresponding_indices.copy()

    #compute the original QR decomposition
    col_pointer = Z_matrix.shape[0] + 1
    random_line = np.random.randn(Z_matrix.shape[0] + 1)

    new_Z_matrix = np.zeros((Z_matrix.shape[0] + 1, Z_matrix.shape[0] + 1))
    new_Z_matrix[:-1, :] = Z_matrix[:, :col_pointer]
    new_Z_matrix[-1, :] = random_line
    QR = np.linalg.qr(new_Z_matrix)
    Q = QR.Q
    R = QR.R

    rhs = np.zeros(new_Z_matrix.shape[0])
    rhs[-1] = 100
    
    for iter in range(max_iter):


        flag = True 
        try:
            delta = np.linalg.lstsq(R, Q.T @ rhs)[0]
        except np.linalg.LinAlgError:
            if verbose:
                print("Caught error. Trying again...")
            Q, R = sp.linalg.qr_delete(Q, R, new_Z_matrix.shape[1]-1, which="col")
            new_col[:-1] = Z_matrix[:, col_pointer]
            new_col[-1] = np.random.randn()
            Q, R = sp.linalg.qr_insert(Q, R, u=new_col, k=new_Z_matrix.shape[1]-1, which="col")
            flag = False

        if flag:
            candidates = eta_vector_copy[:Z_matrix.shape[0] + 1] / delta
            t_star = np.inf
            for j in range(candidates.shape[0]):
                if delta[j] > 0:
                    if candidates[j] < t_star:
                        t_star = candidates[j]
            assert t_star >= 0
        
            eta_vector_copy[:Z_matrix.shape[0] + 1] = eta_vector_copy[:Z_matrix.shape[0]  + 1] - t_star * delta

            if verbose:
                if iter % 100 == 0:
                    print(iter, new_Z_matrix.shape, np.min(eta_vector_copy), eta_vector_copy.shape)
            assert np.min(eta_vector_copy) >= -1e-6

            #index_to_remove = np.argmin(eta_vector_copy)
            index_to_remove = np.argmin(eta_vector_copy[:Z_matrix.shape[0] + 1])
            eta_vector_copy = np.delete(eta_vector_copy, index_to_remove)
            del final_corresponding_indices[index_to_remove]
    
            #delete column in QR decomposition
            Q, R = sp.linalg.qr_delete(Q, R, index_to_remove, which="col")
            #add new column in QR decomposition
            if col_pointer == Z_matrix.shape[1]:
                return Q @ R, eta_vector_copy, final_corresponding_indices
            new_col = np.zeros(new_Z_matrix.shape[0])
            new_col[:-1] = Z_matrix[:, col_pointer]
            new_col[-1] = np.random.randn()
            col_pointer += 1
            Q, R = sp.linalg.qr_insert(Q, R, u=new_col, k=new_Z_matrix.shape[1]-1, which="col")

        
    return new_Z_matrix, eta_vector_copy, final_corresponding_indices



def build_final_solution_exact_caratheodory(y_dic, final_eta_vector, corresponding_indices):
    n = len(y_dic)
    di = y_dic[0][0].shape[0]
    y_final = np.zeros((di, n))
    y_final_max = np.zeros((di, n))
    y_final_sampled = np.zeros((di, n))

    max_coefficient = np.zeros(n)
    coefficients = {}
    

    for index, (i, k) in enumerate(corresponding_indices):
        if final_eta_vector[index] > max_coefficient[i]:
            max_coefficient[i] = final_eta_vector[index]
            y_final_max[:, i] = y_dic[i][0][:, k]
        y_final[:, i] += final_eta_vector[index] * y_dic[i][0][:, k]

        if i in coefficients:
            coefficients[i].append((final_eta_vector[index], k))
        else:
            coefficients[i] = [(final_eta_vector[index], k)]

    for i in range(n):
        alphas = [alpha[0] for alpha in coefficients[i]]
        indices = [alpha[1] for alpha in coefficients[i]]
        li_prime = np.random.choice(indices, p=alphas / np.sum(alphas)) #in theory, we should already have np.sum(alphas) = 1, but we do this for numerical safety
        y_final_sampled[:, i] = y_dic[i][0][:, li_prime]
        
    return y_final, y_final_max, y_final_sampled