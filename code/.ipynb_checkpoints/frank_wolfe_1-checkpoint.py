import numpy as np

def clean_dictionary(y_dic):
    new_dic = {}
    for _, (i, value) in enumerate(y_dic.items()):
        pointer_i = value[1]
        new_array = value[0][:, :pointer_i]
        new_etas = value[2][:pointer_i]
        new_dic[i] = [new_array, new_etas]
    return new_dic  


def frank_wolfe_1(optimal_value_dual, b,
                 K, prob, x_start, verbose=True):

    n = prob.n
    m = prob.m
    di = prob.di # for now we assume that all component dimension d_i are the same 
    #di = x_start.shape[0]
    
    z_star = np.zeros(1 + m)
    z_star[0] = optimal_value_dual
    z_star[1:] = b
    
    
    x_start_col = x_start.reshape((-1, ), order='F')    
    z_k = np.zeros(1 + m)
    z_k[0] = prob.f(x_start)
    A = prob.construct_A_matrix()
    z_k[1:] = A.dot(x_start_col)

    pointer = 0
    y_dic = {}
    for i in range(n):
        y_dic[i] = [np.zeros((di, 1)), 1, np.zeros(K), -1]      
        y_dic[i][0][:, 0] = x_start[:, i]
        y_dic[i][2][0] = 1

    grad_norm_list = []
    g_k = 0
    
    
    for k in range(1, K):
        grad_k = np.clip(z_k - z_star, 0.0, None)
        grad_k_0 = grad_k[0]
        grad_k_rest = grad_k[1:]

        grad_norm = np.linalg.norm(grad_k)
        grad_norm_list.append(grad_norm)

        if verbose:
            if k%100 == 0 or k==0:
                print(f"Iteration {k} : ||z_k - z*||_+^2 = {grad_k_0**2} + { np.linalg.norm(grad_k_rest)**2}")
                print(f"     ||grad(z_k)|| = {grad_norm}")
        
        s_k = np.zeros(1 + m)

        for i in range(n):
            Ai = prob.construct_Ai_matrix(i)
            ATi_dot_gk = Ai.T.dot(grad_k_rest)
            y_ik = prob.lmo_1(i, grad_k_0, ATi_dot_gk)

            f_biconjugate_ik = prob.f_i(i, y_ik)             
            Ati_dot_y_ik = Ai.dot(y_ik)
            s_k += np.concatenate(([f_biconjugate_ik], Ati_dot_y_ik))

            pointer_i = y_dic[i][1]
            #get index of where in the array where the current y_ik is in the dic
            # if it has not been seen before, this will return an empty array
            matching_y_ik = np.where(True == np.all(y_dic[i][0][:, :pointer_i] == y_ik[:, None], axis=0))[0]
            if matching_y_ik.shape[0] == 0:
                y_dic[i][0] = np.concatenate((y_dic[i][0], y_ik.reshape(-1, 1)), axis=1)
                y_dic[i][3] = -1
                y_dic[i][1] += 1
            else:
                y_dic[i][3] = matching_y_ik

        d_k = s_k - z_k
        g_k = -grad_k.dot(d_k)
        etak = min(1, g_k/d_k.dot(d_k))
        assert etak > 0
        
        #now update stepsize accounting
        for i in range(n):
            if y_dic[i][3] == -1:
                #this means we just added a new point
                y_dic[i][2] *= (1-etak)
                y_dic[i][2][y_dic[i][1] - 1] = etak
            else:
                y_dic[i][2] *= (1-etak)
                y_dic[i][2][y_dic[i][3]] += etak
        
        z_k = (1 - etak) * z_k + etak * s_k               


    y_dic = clean_dictionary(y_dic)
            
    return z_k, grad_norm_list, y_dic
