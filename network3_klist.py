import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom

from network0_def import Network
from network1_config import *



def SIR_cik(n, edge_ls, avg_n=100): 
    """Simulate the SIR model on the given network represented by disjoint set.
    To return the average cluster size of one random node after going through k edges.
    avg_n: to average over avg_n number of nodes (with different degrees), assuming each of them as initial seed node, respectively."""

    C = DisjointSet(range(n))
    m = edge_ls.shape[0] # Total number of edges
    seed_node = np.random.randint(n) # Randomly assign seed node
    
    # cluster size containing node i in [0,n) after going through k edges in [0,m]
    ci_k = np.zeros((m,avg_n), dtype=np.int32) 
    all_infection = False
    for k, edge in enumerate(edge_ls):
        if all_infection:
            print("All nodes are infected!!! Exit loop.")
            break
        ci_k_ary = np.zeros(avg_n, dtype=np.int32) # to store the cluster size of each of the first avg_n nodes after going through k edges
        C.merge(edge[0], edge[1])


        for i in range(avg_n):
            # print(f"index:{i}/{avg_n}\r", )
            ci_k_ary[i] = C.subset_size(i)
            if ci_k_ary[i] == n:
                ci_k[k:] = n*np.ones((m-k,avg_n), dtype=np.int32)
                all_infection = True
                print("All nodes are infected.")
                break

        ci_k[k] = ci_k_ary       

        # print(f"edge {k+1}/{m} done: range of ci_k = {min(ci_k[k])}, {max(ci_k[k])}.")

    # inital cluster size is 1 for each seed node (i.e. concatenate with a row of 1s)
    ci_k = np.concatenate((np.ones((1,avg_n), dtype=np.int32), ci_k), axis=0) 
    print("shape of ci_k:", ci_k.shape) # shape(m+1,avg_n)

    return ci_k # shape(m+1,avg_n)


def SIR_bino_coeff(edge_ls, lambda_ary):
    """Compute the binomial coefficient for each lambda value and each number of edges infected."""
    m = edge_ls.shape[0] # Total number of edges
    k_ary = np.arange(m+1, dtype=np.int32) # number of edges infected
    lam_n = len(lambda_ary) # number of lambda values
    bino_coeff = np.zeros((lam_n, m+1), dtype=np.float32) # store the binomial coefficients for each lambda
    for idx, lambda_ in enumerate(lambda_ary):
        bino_coeff[idx] = binom.pmf(k_ary, m, lambda_) # shape(lam_n, m+1)
    
    # check the shape of the coefficient array is (lam_n, m+1)
    print("shape of bino coeff:", bino_coeff.shape)
    print(bino_coeff)

    return bino_coeff # shape(lam_n, m+1)



def SIR_ci_lambda(n, edge_ls, lambda_ary, avg_n=100, compute_mean=True):
    """Compute average final cluster size containing a random seed node (aftering looping over all edges) for each lambda value.
    by dot product of binomial coefficient and cluster size array (as a function of k)."""

    ci_lambda = np.matmul(SIR_bino_coeff(edge_ls, lambda_ary), SIR_cik(n, edge_ls, avg_n=avg_n))

    
    # each row is the cluster sizes of first avg_n nodes for the particular lambda corresponding to that row
    print("shape of all ci_lambda (not averaged):", ci_lambda.shape) # shape (lam_n, n_avg) 

    if compute_mean:
        # average over all avg_n nodes to get the average cluster size for each lambda (for a random node)
        ci_lambda = np.mean(ci_lambda, axis=1) # shape (lam_n,)

    return ci_lambda 
