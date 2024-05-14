import numpy as np
import matplotlib.pyplot as plt
import timeit

from network_def import Network
from wk1_sampling import *


def time_compare():
    # run both algorithms to generate G(n, 10/(n-1)) with n = 64, 128, ..., 1024 nodes
    n_ary = np.array([2**i for i in range(6, 11)])

    t_ary_1 = np.empty(len(n_ary))
    t_ary_2 = np.empty(len(n_ary))
    for i, n in enumerate(n_ary):
        p = 10/(n-1)
        t_ary_1[i] = timeit.timeit(lambda: rm_graph_gen(n, p, True), number=10)
        t_ary_2[i] = timeit.timeit(lambda: rm_graph_gen2(n, p, True), number=10)
    print("time for naive sampling algorithm:")
    print(t_ary_1)
    print(f"time for 2-stage algorithm:")
    print(t_ary_2)



    # Perform linear regression in log scale
    coeff_1= np.polyfit(np.log(n_ary), np.log(t_ary_1), deg=1)
    t_1_fit = np.poly1d(coeff_1) 
    coeff_2= np.polyfit(np.log(n_ary), np.log(t_ary_2), deg=1)
    t_2_fit = np.poly1d(coeff_2) 

    # plot the results in log scale
    plt.loglog(n_ary, t_ary_1, "b", label="naive sampling")
    plt.loglog(n_ary, np.exp(t_1_fit(np.log(n_ary))), "--b", label=f"naive slope: {coeff_1[0]:.2f}")
    plt.loglog(n_ary, t_ary_2, "g", label="2-stage sampling")
    plt.loglog(n_ary, np.exp(t_2_fit(np.log(n_ary))), "--g", label=f"2-stage slope: {coeff_2[0]:.2f}")


    # plt.plot(n_ary, t_ary_1, label="naive sampling")
    # plt.plot(n_ary, t_ary_2, label="2-stage sampling")
    plt.xlabel("log n")
    plt.ylabel("log time(s)")
    plt.legend()
    plt.show()

    return t_ary_1, t_ary_2





