import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom

from wk2_1_config_model import *
from network_def import Network
from wk3_4_SIR_djset import *


def SIR_cik(n, edge_ls, single_node = False):
    """Simulate the SIR model on the given network represented by disjoint set.
    Assume single seed node to be node 0."""

    C = DisjointSet(range(n))
    m = edge_ls.shape[0] # Total number of edges
    seed_node = np.random.randint(n) # Randomly assign seed node
    
    # cluster size containing node i in [0,n) after going through k edges in [0,m]
    ci_k = np.zeros((m,n)) 
    for k, edge in enumerate(edge_ls):
        ci_k_ary = np.zeros(n) # store the cluster size of each node after going through k edges
        C.merge(edge[0], edge[1])

        for subset in C.subsets():
            subset_size = len(subset)
            if subset_size == n: # if the subset is the whole network, then break
                ci_k[k:] = n*np.ones((m-k,n))
                break
            ci_k_ary[subset] = subset_size
        ci_k[k] = ci_k_ary       

        print(f"edge {k+1}/{m} done: ci_k = {ci_k[k]}")

    # inital cluster size is 1 for each seed node (i.e. concatenate with a row of 1s)
    ci_k = np.concatenate((np.ones((1,n)), ci_k), axis=0) 
    print("shape of ci_k:", ci_k.shape)

    if single_node: # take the column of the matrix that corresponds to the seed node
        # shape(m+1,1)
        return ci_k[:,seed_node] # FIXME: or should we fix the seed node to always the first node???
    else:
        # shape(m+1,n)
        return ci_k


def SIR_bino_coeff(edge_ls, lambda_ary):
    
    m = edge_ls.shape[0] # Total number of edges
    k_ary = np.arange(m+1) # number of edges infected
    lam_n = len(lambda_ary) # number of lambda values
    bino_coeff = np.zeros((lam_n, m+1)) # store the binomial coefficients for each lambda
    for idx, lambda_ in enumerate(lambda_ary):
        bino_coeff[idx] = binom.pmf(k_ary, m, lambda_) # shape(lam_n, m+1)
    
    # check the shape of the coefficient array is (lam_n, m+1)
    print("shape of bino coeff:", bino_coeff.shape)
    print(bino_coeff)
    # shape(lam_n, m+1)
    return bino_coeff

def SIR_ci_lambda(n, edge_ls, lambda_ary, single_node = False):
    ci_lambda = np.matmul(SIR_bino_coeff(edge_ls, lambda_ary), SIR_cik(n, edge_ls, single_node))

    # shape (lam_n, n) 
    # each row is the cluster sizes of all nodes for the particular lambda corresponding to that row
    print("shape of ci_lambda:", ci_lambda.shape)

    return ci_lambda


if __name__ == "__main__":
    n = 10000
    mean = 20
    lambda_ary = np.linspace(0.00, 0.15, 16) 
    iter_n = 10
    mu = np.zeros(len(lambda_ary)) # Mean of the number of total infections mu over lambda
    cov = np.zeros(len(lambda_ary)) # coefficient of variation of the number of total infections sigma / mu over lambda

    # Generate a new network for each iteration, and calculate the number of total infections for each lambda
    output = np.empty(iter_n, dtype=object) # Store the number of total infections for each iteration and each lambda
    # generate one single network for all iterations
    edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
    
    
    # Run for one iteration only, take mean and cov over all nodes
    edge_ls = np.random.permutation(edge_ls) # generate an edge list
    output= SIR_ci_lambda(n, edge_ls, lambda_ary) # output shape (lam_n, n)
    print("output matrix:", output) 
    # Compute mean and coefficient of variation over all nodes
    mu = np.mean(output, axis=1)        # shape (lam_n,)
    cov = np.std(output, axis=1) / mu   # shape (lam_n,)
    
    print("mu:", mu)
    print("cov:", cov)
    plt.figure()
    plt.errorbar(lambda_ary, mu, yerr=cov, fmt='o')
    plt.xlabel("Transmission rate")
    plt.ylabel("Average number of total infections")
    plt.savefig("wk3_5_SIR_cik.png")
    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    # for itn in range(iter_n):
    #     print("iter: ", itn, '/', iter_n)
    #     edge_ls = np.random.permutation(edge_ls) # shuffle the edge list
    #     output[itn] = SIR_ci_lambda(n, edge_ls, lambda_ary) # output shape (iter_n, lam_n, n)
    #     print("output matrix:", output[itn]) # shape (lam_n, n)
    

        
    # # Compute mean and coefficient of variation over all iterations
    # mu = np.mean(output, axis=0)        # shape (lam_n, n)
    # cov = np.std(output, axis=0) / mu   # shape (lam_n, n)
        

    # print("mu:", mu.shape)
    # print("cov:", cov.shape)
    
    # # Write the output to a text file
    # f = open('wk3_5_SIR_cik.txt', mode='a')
    # for itn in range(iter_n):
    #     f.write(f'Iteration {itn+1}:\n')
    #     f.write(f'Output matrix: {output[itn]}.\n')

    # f.write(f'Mean of the number of total infections mu over lambda: {mu}.\n')
    # f.write(f'Coefficient of variation of the number of total infections sigma / mu over lambda: {cov}.\n')
    # f.close()

