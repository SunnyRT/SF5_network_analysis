import numpy as np
import matplotlib.pyplot as plt
import timeit

from network_def import Network
from wk1_1_naive_sampling import rm_graph_gen
from wk1_5_eff_sampling import rm_graph_gen2

#FIXME: not working properly!!!!!!!!
def time_compare():
    # run both algorithms to generate G(n, 10/(n-1)) with n = 64, 128, ..., 1024 nodes
    # n_ary = np.array([2^i for i in range(6, 11)])
    n_ary = np.array([1024])

    t_ary_1 = np.empty(len(n_ary))
    t_ary_2 = np.empty(len(n_ary))
    for i, n in enumerate(n_ary):
        p = 10/(n-1)
        t_ary_1[i] = timeit.timeit(lambda: rm_graph_gen(n, p), number=100)
        t_ary_2[i] = timeit.timeit(lambda: rm_graph_gen2(n, p), number=100)
    print("time for naive sampling algorithm:")
    print(t_ary_1)
    print(f"time for 2-stage algorithm:")
    print(t_ary_2)

    # plot the results in log scale
    plt.plot(n_ary, np.log(t_ary_1), label="naive sampling")
    plt.plot(n_ary, np.log(t_ary_2), label="2-stage sampling")
    plt.xlabel("n")
    plt.ylabel("time (s)")
    plt.legend()
    plt.show()





