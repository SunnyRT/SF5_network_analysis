import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import DisjointSet
from scipy.special import comb

from wk2_1_config_model import *
from network_def import Network
from wk3_4_SIR_djset import *


def SIR_cik(edge_ls, lambda_ary):
    """Simulate the SIR model on the given network represented by disjoint set.
    Assume single seed node to be node 0."""

    C = DisjointSet(range(n))
    m = edge_ls.shape[0] # Total number of edges
    
    # cluster size containing node 0 after going through k edges
    ci_k = np.zeros(m) 
    for k, edge in enumerate(edge_ls):
        C.merge(edge[0], edge[1])

        # compute cluster size of the seed node (assume node 0)
        ci_k[k] = C.subset_size(0)

    # inital cluster size is 1 (i.e. seed node when k=0)
    ci_k = np.insert(ci_k, 0, 1) 

    # Final cluster size after going through m edges for each lambda
    ci_lambda = np.zeros(len(lambda_ary))
    for idx, lambda_ in enumerate(lambda_ary):
        ci_lambda[idx] = sum(comb(m,k)*lambda_**k*(1-lambda_)**(m-k)*ci_k[k] for k in range(m+1))
   
    return ci_lambda

if __name__ == "__main__":
    n = 10000
    mean = 20
    lambda_ary = np.linspace(0.00, 0.15, 16) 
    iter_n = 100
    mu = np.zeros(len(lambda_ary)) # Mean of the number of total infections mu over lambda
    cov = np.zeros(len(lambda_ary)) # coefficient of variation of the number of total infections sigma / mu over lambda

    # Generate a new network for each iteration, and calculate the number of total infections for each lambda
    output = np.empty(iter_n, dtype=object) # Store the number of total infections for each iteration and each lambda
    for itn in range(iter_n):
        print("iter: ", itn, '/', iter_n)
        edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
        output[itn] = SIR_cik(edge_ls, lambda_ary)

    # Compute mean and coefficient of variation over all iterations
    mu = np.mean(output, axis=0)
    cov = np.std(output, axis=0) / mu

    plt.figure()
    plt.errorbar(lambda_ary, mu, yerr=cov, fmt='o')
    plt.xlabel("Transmission rate")
    plt.ylabel("Average number of total infections")
    plt.savefig("wk3_5_SIR_cik.png")
    plt.show()