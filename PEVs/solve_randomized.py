import pulp
import numpy as np


def solve_randomized(opt_dual_value, pev_prob, solve_contracted=True):
    n = pev_prob.n
    N = pev_prob.m

    w = np.random.randn(n, N)

    prob = pulp.LpProblem("PEV", pulp.LpMinimize)
    
    #variables for the relaxed LP problem
    u_vars = pulp.LpVariable.dicts(
            "w", (range(n), range(N)), lowBound=0, upBound=1, cat="Continuous"
        )

    prob += (pulp.lpSum([w[i, j] * u_vars[i][j] for i in range(n) for j in range(N)]), "Total cost")
    #prob += (pulp.lpSum([P[i] * ( Cu[j] * u_vars[i][j] for j in range(m)) for i in range(n)]), "Total cost")
    prob += pulp.lpSum([pev_prob.P[i] * pev_prob.Cu[j] * u_vars[i][j] for j in range(N) for i in range(n)]) <= opt_dual_value, "Bidual constraints"

    for j in range(N):
        if solve_contracted:
            prob += pulp.lpSum([pev_prob.P[i] * u_vars[i][j] for i in range(n)]) <= pev_prob.P_max_bar[j], (f"Coupling constraint max at time step {j}")
        else:
            prob += pulp.lpSum([pev_prob.P[i] * u_vars[i][j] for i in range(n)]) <= pev_prob.P_max[j], (f"Coupling constraint max at time step {j}")
            #prob += pulp.lpSum([pev_prob.P[i] * u_vars[i][j] for i in range(n)]) >= pev_prob.P_min[j], (f"Coupling constraint min at time step {j}")
            #no need to consider P_min in charging only scenario when P_min < 0

    for i in range(n):
        prob += pulp.lpSum([u_vars[i][j] for j in range(N)]) >= np.ceil((pev_prob.E_ref[i] - pev_prob.E_init[i]) / (pev_prob.P[i] * pev_prob.delta_T * pev_prob.xi_u[i])), (f"E_ref constraint at {i}")
        prob += pulp.lpSum([u_vars[i][j] for j in range(N)]) <= np.ceil((pev_prob.E_max[i] - pev_prob.E_init[i]) / (pev_prob.P[i] * pev_prob.delta_T * pev_prob.xi_u[i])), (f"E_max constraint at {i}")
    
    
    
    res = prob.solve()
    u = np.zeros((N, n))
    for i in range(n):
        for j in range(N):
            u[j, i] = u_vars[i][j].varValue
    u_col = u.reshape((-1,), order='F')
    return u, res


def build_final_solution_randomized(u, prob):
    u_final = np.zeros_like(u)
    for i in range(prob.n):
        non_trivial_CC = np.where(np.logical_and(u[:, i]>0, u[:, i]<1))[0]
        sum_nb_non_trivial_CC = round(np.sum((u[:, i])[non_trivial_CC]))
        if sum_nb_non_trivial_CC > 0:
            sorted_indices = np.argsort((u[:, i]))[::-1]

            nb_1 = sum_nb_non_trivial_CC
            counter = 0
            for j in sorted_indices:
                if u[j, i] == 1:
                    u_final[j, i] = 1
                elif u[j, i] > 0:
                    if counter < nb_1:
                        u_final[j, i] = 1
                        counter += 1
                    else:
                        u_final[j, i] = 0
                else:
                    u_final[j, i] = 0
        else:
            u_final[:, i] = u[:, i]
    return u_final 