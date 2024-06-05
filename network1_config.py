import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial


from network0_def import Network

def config_graph_gen(n, k_ary, m_only=False):
    """Generate a configuration model network with n nodes and degree array k_ary."""
    S = np.array([i for i in range(n) for _ in range(k_ary[i])])
    S = np.random.permutation(S)
    if len(S) % 2 == 1:
        S = S[:-1]

    # S gives list of edges, with each row representing the index of the two nodes connected by the edge
    S = S.reshape(-1, 2)
    adj_m = np.zeros((n, n), dtype=int)
    for idx in S:
        if idx[0] != idx[1]:
            adj_m[idx[0]][idx[1]] = 1
            adj_m[idx[1]][idx[0]] = 1

    if m_only:
        return adj_m
    else:
        rm_graph = Network(adj_m=adj_m)
        return rm_graph


def deg_dist_poisson(n, mean):
    """Generate a degree array k_ary for n nodes from Poisson distribution with mean = lam."""
    lam = float(mean)
    k_ary = np.random.poisson(lam, n)
    return k_ary


def deg_dist_geo(n,mean):
    """Generate a degree array k_ary for n nodes from geometric distribution (# failures before success) with mean = (1-p)/p."""
    p = 1/(1+mean)
    k_ary = np.random.geometric(p, n)-1 # subtract 1 to account for distribution from np.random.geometric which starts from 1
    return k_ary

