import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


from network0_def import Network
from network1_config import *
from network2_clustersize import *


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

    return s # shape (n,)

def crit_lambda(network):
    A = network.adj_m
    A = sp.csr_matrix(A, dtype=float)
    # compute max eigenvalue of adjacency matrix A
    v_max = eigsh(A, k=1, which='LM', return_eigenvectors=False)[0] # "k=1" specifies largest eigenvalue, "LM" means "Largest Magnitude
    # compute the critical lambda
    lambda_crit = 1/v_max
    return lambda_crit


