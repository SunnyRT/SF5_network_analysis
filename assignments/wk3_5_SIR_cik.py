import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom

from network_def import Network
from wk2_1_config_model import *
from wk3_4_SIR_djset import *


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


# if __name__ == "__main__":
#     n = 10000
#     mean = 20
#     lambda_ary = np.linspace(0.00, 0.15, 16) 
#     avg_n = 100
#     mu = np.zeros(len(lambda_ary)) # Mean of the number of total infections mu over lambda
#     cov = np.zeros(len(lambda_ary)) # coefficient of variation of the number of total infections sigma / mu over lambda

#     # Generate a new network for each iteration, and calculate the number of total infections for each lambda
#     # output = np.empty(iter_n, dtype=object) # Store the number of total infections for each iteration and each lambda
#     # generate one single network for all iterations
#     edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
    
    
#     # Run for one iteration only, take mean and cov over first 100 nodes
#     output= SIR_ci_lambda(n, edge_ls, lambda_ary, avg_n, False) # output shape (lam_n, avg_n)
#     print("output matrix:", output) 
#     # Compute mean and coefficient of variation over all avg_n nodes
#     mu = np.mean(output, axis=1)        # shape (lam_n,)
#     cov = np.std(output, axis=1) / mu   # shape (lam_n,)
    
#     print("mu:", mu)
#     print("cov:", cov)
#     plt.figure()
#     plt.errorbar(lambda_ary, mu, yerr=cov, fmt='o')
#     plt.xlabel("Transmission rate")
#     plt.ylabel("Average number of total infections")
#     plt.savefig("wk3_5_SIR_cik.png")
#     plt.show()    


if __name__ == "__main__":
    n = 4096
    mean = 10 # mean degree
    lambda_ary = np.linspace(0.00, 0.5, 26)
    avg_n = 1000
    mu_p = np.zeros(len(lambda_ary)) # Mean of the number of total infections mu over lambda
    mu_g = np.zeros(len(lambda_ary)) 
    cov_p = np.zeros(len(lambda_ary)) # coefficient of variation of the number of total infections sigma / mu over lambda
    cov_g = np.zeros(len(lambda_ary))

    # Generate a new network for each iteration, and calculate the number of total infections for each lambda
    # output = np.empty(iter_n, dtype=object) # Store the number of total infections for each iteration and each lambda
    # generate one single network for all iterations
    edge_ls_p = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
    edge_ls_g = config_graph_edge_ls(n, deg_dist_geo(n, mean))
    
    
    # Run for one iteration only, take result for first node only (for avg_n=1 )
    output_p= SIR_ci_lambda(n, edge_ls_p, lambda_ary, avg_n, False) # output shape (lam_n, avg_n)
    output_g= SIR_ci_lambda(n, edge_ls_g, lambda_ary, avg_n, False) # output shape (lam_n, avg_n)
    print("output matrix p:", output_p) 
    print("output matrix g:", output_g)
    # Compute mean and coefficient of variation over all avg_n nodes
    mu_p = np.mean(output_p, axis=1)        # shape (lam_n,)
    cov_p = np.std(output_p, axis=1) / mu_p   # shape (lam_n,)
    mu_g = np.mean(output_g, axis=1)        # shape (lam_n,)
    cov_g = np.std(output_g, axis=1) / mu_g   # shape (lam_n,)
    
    print("mu:", mu_p)
    print("cov:", cov_p)
    fig, ax1 = plt.subplots()

    # Plot data on the first y-axis
    ax1.plot(lambda_ary, mu_p, "o-", color = 'tab:blue', label="p: μ")
    ax1.plot(lambda_ary, mu_g, "^-", color = 'tab:orange', label="p: μ")
    ax1.set_xlabel('Transmission rate')
    ax1.set_ylabel("Mean")
    ax1.tick_params(axis='y')

    # Create the second y-axis
    ax2 = ax1.twinx()
    ax2.plot(lambda_ary, cov_p, "--", color = 'tab:blue', label="p: cov")
    ax2.plot(lambda_ary, cov_g, "--", color = 'tab:orange', label="p: cov")
    ax2.set_ylabel("Coefficient of variation")
    ax2.tick_params(axis='y')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
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

