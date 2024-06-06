import numpy as np
import matplotlib.pyplot as plt
import timeit

from network_def import Network
from wk2_1_config_model import *
from wk3_2_crit_thres import SIR_Nr_lam
from wk3_4_SIR_djset import SIR_djset
from wk3_5_SIR_cik import SIR_ci_lambda


def time_compare():
    # run both algorithms to generate G(n, 10/(n-1)) with n = 64, 128, ..., 1024 nodes
    n_ary = np.array([2**i for i in range(6, 11)])
    lambda_ary = np.array([0.2])
    lambda_ = 0.2



    t_ary_1 = np.empty(len(n_ary))
    t_ary_2 = np.empty(len(n_ary))
    t_ary_3 = np.empty(len(n_ary))




    # generate the same configuration model graph for all 3 algorithms
    
    for i, n in enumerate(n_ary):
        network = config_graph_gen(n, deg_dist_poisson(n, n/64))
        edge_ls = np.array(network.edge_list())
        t_ary_1[i] = n* timeit.timeit(lambda: SIR_Nr_lam(network, 1, lambda_ary, iter_n = 1), number=100)
        t_ary_2[i] = timeit.timeit(lambda: SIR_djset(n, edge_ls, lambda_), number=100)
        t_ary_3[i] = timeit.timeit(lambda: SIR_ci_lambda(n, edge_ls, lambda_ary, avg_n=1, compute_mean=False), number=100)

    print("time for naive simulation algorithm:")
    print(t_ary_1)
    print(f"time for disjoint set algorithm:")
    print(t_ary_2)
    print(f"time for k-list algorithm:")
    print(t_ary_3)



    # Perform linear regression in log scale
    coeff_1= np.polyfit(np.log(n_ary), np.log(t_ary_1), deg=1)
    t_1_fit = np.poly1d(coeff_1) 
    coeff_2= np.polyfit(np.log(n_ary), np.log(t_ary_2), deg=1)
    t_2_fit = np.poly1d(coeff_2) 
    coeff_3= np.polyfit(np.log(n_ary), np.log(t_ary_3), deg=1)
    t_3_fit = np.poly1d(coeff_3)

    # plot the results in log scale
    plt.loglog(n_ary, t_ary_1, "b", label="naive simulation")
    plt.loglog(n_ary, np.exp(t_1_fit(np.log(n_ary))), "--b", label=f"slope: {coeff_1[0]:.2f}")
    plt.loglog(n_ary, t_ary_2, "g", label="disjoint set")
    plt.loglog(n_ary, np.exp(t_2_fit(np.log(n_ary))), "--g", label=f"slope: {coeff_2[0]:.2f}")
    plt.loglog(n_ary, t_ary_3, "r", label="k-list")
    plt.loglog(n_ary, np.exp(t_3_fit(np.log(n_ary))), "--r", label=f"slope: {coeff_3[0]:.2f}")


    # plt.plot(n_ary, t_ary_1, label="naive sampling")
    # plt.plot(n_ary, t_ary_2, label="2-stage sampling")
    plt.xlabel("log n")
    plt.ylabel("log time(s)")
    plt.legend()
    plt.show()

    return t_ary_1, t_ary_2, t_ary_3





