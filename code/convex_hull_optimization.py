import numpy as np
from qpsolvers import solve_qp
from simplex_projection import *


def optimize_over_convex_hull(V_matrix, z_star, solver="proxqp", tol_gt=1e-4, max_iter=np.inf, verbose=False):
    if V_matrix.shape[1] == 1:
        return V_matrix[:, 0], np.array([1])

    if solver == "FW":
        V_matrixT_V_matrix = V_matrix.T @ V_matrix
        V_matrixT_z_star = V_matrix.T @ z_star
        L = np.linalg.norm(V_matrixT_V_matrix, 2)
        lambda_t = np.random.rand(V_matrix.shape[1])
        lambda_t = lambda_t / np.sum(lambda_t)
        g_t = np.inf
        t = 0
        while np.linalg.norm(g_t) > tol_gt and t <= max_iter:
            grad_t = V_matrixT_V_matrix @ lambda_t - V_matrixT_z_star
            min_index = np.argmin(grad_t)
            
            d_t = - lambda_t
            d_t[min_index] += 1
            g_t = -grad_t.dot(d_t)
            #if k > -1:
            eta_t = min(1, g_t/(L * d_t.dot(d_t)))
            assert eta_t > 0
            lambda_t = (1 - eta_t) * lambda_t
            lambda_t[min_index] += eta_t
            t += 1
        
        opt_lambda = lambda_t

    elif solver == "PGD":
        V_matrixT_V_matrix = V_matrix.T @ V_matrix
        V_matrixT_z_star = V_matrix.T @ z_star
        L = np.linalg.norm(V_matrixT_V_matrix, 2)
        lambda_t = np.random.rand(V_matrix.shape[1])
        lambda_t = lambda_t / np.sum(lambda_t)
        stepsize = 1/L
        g_t = np.inf
        t = 0
        while np.linalg.norm(g_t) > tol_gt and t <= max_iter:
            grad_t = V_matrixT_V_matrix @ lambda_t - V_matrixT_z_star
            lambda_t = euclidean_proj_simplex(lambda_t - stepsize * grad_t)

            min_index = np.argmin(grad_t)
            d_t = - lambda_t
            d_t[min_index] += 1
            g_t = -grad_t.dot(d_t)
            t += 1
                
        opt_lambda = lambda_t
        
    else:    
        Q_matrix = V_matrix.T @ V_matrix
        q_vector = -V_matrix.T @ z_star
        A = np.ones((1, V_matrix.shape[1]))
        b = np.ones(1)
        opt_lambda = solve_qp(P=Q_matrix, q=q_vector, A=A, b=b, lb=np.zeros(V_matrix.shape[1]), solver=solver, verbose=verbose)
        try:
            return V_matrix @ opt_lambda, opt_lambda
        except:
            opt_lambda = solve_qp(P=Q_matrix, q=q_vector, A=A, b=b, lb=np.zeros(V_matrix.shape[1]), solver="cvxopt", verbose=verbose) #more robust
    return V_matrix @ opt_lambda, opt_lambda