import numpy as np
import matplotlib.pyplot as plt

from wk2_1_config_model import *
from wk4_vaccine import *


def compute_s_prob_adjls(edge_ls, lambda_, n, tol=1e-6, max_iter=1000):
    adj_ls = edge_to_adj_ls(edge_ls, n)
    
    s = np.random.rand(n)
    s_prev = np.zeros(n)

    for _ in range(max_iter):
        if np.allclose(s, s_prev, atol=tol):
            break
        s_prev = s.copy()

        for i in range(n):
            s_j = np.array([s_prev[j] for j in adj_ls[i]])
            log_terms = np.log(1 - lambda_ + s_j * lambda_)
            s[i] = np.exp(np.sum(log_terms))

    return s


if __name__ == "__main__":

    n = 10000
    mean = 20
    v = 0.4
    lam_ary = np.linspace(0, 0.3, 30)
    avg_n = 1000

    # empirical simulation
    edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
    edge_ls_vi = vaccine_edgels(edge_ls, v, n) # remove edges connected to vaccinated nodes
    
    edge_ls = np.random.permutation(edge_ls) # reshuffle the edge list
    edge_ls_vj = vaccine_edgels(edge_ls, v, n, vacc_friend=True) # remove edges connected to nominated friends

    output_nodes = SIR_ci_lambda(n, edge_ls_vi, lam_ary, avg_n, compute_mean=True) # output shape (lam_n,)
    output_friends = SIR_ci_lambda(n, edge_ls_vj, lam_ary, avg_n, compute_mean=True) # output shape (lam_n,)

    print("Empirical simulation done.")

    # prediction from si_prob
    s_ary_vi = np.zeros(len(lam_ary))
    s_ary_vj = np.zeros(len(lam_ary))

    for idx, lambda_ in enumerate(lam_ary):
        s_vi = compute_s_prob_adjls(edge_ls_vi, lambda_, n)
        s_vj = compute_s_prob_adjls(edge_ls_vj, lambda_, n)
        s_ary_vi[idx] = np.mean(s_vi)
        s_ary_vj[idx] = np.mean(s_vj)

    Nr_ary_vi = (1 - s_ary_vi)*n # output shape (lam_n,)
    Nr_ary_vj = (1 - s_ary_vj)*n # output shape (lam_n,)
    
    print("Theoretical prediction from si probability done.")


    # outputs shape (v_n, lam_n)
    # plot the results
    plt.figure()
    plt.plot(lam_ary, output_nodes, "o-", color='tab:blue', label="node (simulation)")
    plt.plot(lam_ary, Nr_ary_vi, "--", alpha = 0.5, color='tab:blue', label="node (si approx.)")
    plt.plot(lam_ary, output_friends, "o-", color='tab:orange', label="friends (simulation)")
    plt.plot(lam_ary, Nr_ary_vj, "--", alpha = 0.5, color='tab:orange', label="friends (si approx.)")
    plt.xlabel("Transmission rate")
    plt.ylabel("Average number of total infections")
    plt.legend()
    plt.show()


