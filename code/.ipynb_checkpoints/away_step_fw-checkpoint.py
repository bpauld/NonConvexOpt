def frank_wolfe_2_away_steps(z_K, T, y_dic, prob, f_values=None, tol1=1e-2, tol2=1e-9, verbose=True,
                             nb_non_trivial_CC_wanted="default"):
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
    opt_lambda = [1]

    nb_FW_steps = 0
    nb_A_steps = 0
    nb_reset_FW = 0

    
    for t in range(1, T):       
        grad_t = z_t - z_star

        grad_list.append(np.linalg.norm(grad_t))

        #now find the smallest among all the candidates
        best_i, best_k = lmo_2(grad_t, y_dic, f_values, prob)
        print(best_i, best_k)
        s_t = build_vector_from_indices(best_i, best_k, y_dic, prob)

        newly_added_i = best_i

        #solve with proxqp
        d_FW = s_t - z_t
        v_t, index_v_t = find_argmax_in_active_set(grad_t, active_set, y_dic, prob)
        d_A = z_t - v_t

        FW_step = True
        if - grad_t.dot(d_FW) >= - grad_t.dot(d_A):
            nb_FW_steps += 1
            d_t = d_FW 
            gamma_max = 1
        else:
            nb_A_steps += 1
            FW_step = False
            d_t = d_A
            gamma_max = opt_lambda[index_v_t] / (1 - opt_lambda[index_v_t])

        gamma_t = min(gamma_max, - np.dot(z_t - z_star, d_t) / d_t.dot(d_t))
        #print(gamma_max, FW_step, active_set, opt_lambda)
        assert gamma_t > 0

        z_t = z_t + gamma_t * d_t

        #gamma_t_prime = 2 /(t + 2)
        #z_t = (1 - gamma_t_prime) * z_t + gamma_t_prime * s_t

        if FW_step:
            if gamma_t == 1:
                nb_reset_FW += 1
                #remove active set
                active_set = [(best_i, best_k)]
                opt_lambda = [1]
                is_index_added = np.zeros(n)
                is_index_added[best_i] = 1
            else:
                if (best_i, best_k) in active_set:
                    #find the index where it is
                    index_s_t = active_set.index((best_i, best_k))
                    opt_lambda = np.array(opt_lambda)
                    opt_lambda *= (1 - gamma_t)
                    opt_lambda[index_s_t] += gamma_t
                    opt_lambda = opt_lambda.tolist()
                else:
                    active_set.append((best_i, best_k))
                    opt_lambda = (np.array(opt_lambda) * (1 - gamma_t)).tolist()
                    opt_lambda.append(gamma_t)
                    is_index_added[best_i] = 1
        else:
            if gamma_t == gamma_max:
                del active_set[index_v_t]
                del opt_lambda[index_v_t]
                opt_lambda = np.array(opt_lambda)
                opt_lambda *= (1 + gamma_t)
                opt_lambda = opt_lambda.tolist()
            else:
                opt_lambda = np.array(opt_lambda)
                opt_lambda *= (1 + gamma_t)
                opt_lambda[index_v_t] -= gamma_t
                opt_lambda = opt_lambda.tolist()
                
                
        
        

        
        matching_newly_added_i = np.array([(j, opt_lambda[j]) for j, tuple_ik in enumerate(active_set) if tuple_ik[0]==newly_added_i])
        nb_index_added = np.sum(is_index_added)
        if verbose:
            frequency = T / 10
            if t % frequency == 0:
                print(f"At iteration {t}, grad_norm = {np.linalg.norm(grad_t)}, nb_index_added = {nb_index_added}, len(active_set) = {len(active_set)}")
                print(f"Number FW steps = {nb_FW_steps}, Number away steps = {nb_A_steps}, Number reset FW = {nb_reset_FW}")
                print(active_set)


        #if nb_missing_indices == 0 and nb_non_trivial_CC == nb_non_trivial_CC_wanted:
        #    break

    return z_t, np.array(active_set), grad_list, opt_lambda, is_index_added
