import numpy as np

def solve_dual_gd(prob, eta="1/k", max_iter=1000, solve_contracted_problem=False, verbose=True):

    lambda_k = np.zeros(prob.m)
    fmax = -np.inf

    if solve_contracted_problem:
        b = prob.b_bar.copy()
    else:
        b = prob.b.copy()
    
    for k in range(max_iter):
        grad_k = b.copy()
        fk = b.dot(lambda_k)

        for i in range(prob.n):
            #Ai = prob.construct_Ai_matrix(i)
            #fi, gi = prob.f_conjugate_i(i, - Ai.T.dot(lambda_k))
            fi, gi = prob.f_conjugate_i(i, - prob.compute_AiT_dot_g(i, lambda_k))
            
            grad_k += - prob.compute_Ai_dot_y(i, gi) 
            fk += fi

        if eta=="1/k":
            etak = 1/(k+1)
        else:
            etak = eta
        lambda_k = lambda_k - etak * grad_k
        lambda_k = np.clip(lambda_k, 0, None)
        if verbose:
            if k%100 == 0:
                print("    ", k, -fk, np.linalg.norm(grad_k), np.linalg.norm(lambda_k))

        fmax = -fk if -fk > fmax else fmax

    return fmax

            