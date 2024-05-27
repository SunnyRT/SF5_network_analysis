import numpy as np
import matplotlib.pyplot as plt

from wk2_1_config_model import *
from network_def import Network

def compute_s_prob(network, lambda_, tol=1e-6, max_iter=1000):
    n = network.n
    adj_ls = network.adj_ls
    
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

    return s_prev, s


if __name__ == "__main__":
    n = 10000
    mean = 20
    lambda_ary = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    s_ary_p = np.zeros(len(lambda_ary))
    s_ary_g = np.zeros(len(lambda_ary))

    # Generate configuration model network with Poisson degree distribution
    network_p = config_graph_gen(n, deg_dist_poisson(n, mean))
    for idx, lambda_ in enumerate(lambda_ary):
        _, s = compute_s_prob(network_p, lambda_)
        s_ary_p[idx] = np.average(s)
        print(f"Poisson: lambda = {lambda_}, s = {np.average(s)}")
    Nr_ary_p = n * (1 - s_ary_p)

    # Generate configuration model network with geometric degree distribution
    network_g = config_graph_gen(n, deg_dist_geo(n, mean))
    for idx, lambda_ in enumerate(lambda_ary):
        _, s  = compute_s_prob(network_g, lambda_)
        s_ary_g[idx] = np.average(s)
        print(f"Geometric: lambda = {lambda_}, s = {np.average(s)}")
    Nr_ary_g = n * (1 - s_ary_g)

    # plt.ion()
    # img = plt.imread("wk3_2_plotlog.png")
    # plt.imshow(img)

    plt.figure()
    plt.plot(lambda_ary, Nr_ary_p, "--", label="Poisson prediction")
    plt.plot(lambda_ary, Nr_ary_g, "--", label="Geometric prediction")
    plt.xlabel("Transmission rate")
    plt.ylabel("Predicted number of total infections")
    plt.legend()
    plt.savefig("wk3_3_SIR_si.png")
    plt.show()