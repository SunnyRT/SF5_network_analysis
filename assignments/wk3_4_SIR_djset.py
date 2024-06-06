import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import DisjointSet

from network_def import Network
from wk2_1_config_model import *

def config_graph_edge_ls(n, k_ary):
    """Generate a configuration model edge list with n nodes and degree array k_ary."""
    edge_ls = np.array([i for i in range(n) for _ in range(k_ary[i])])
    edge_ls = np.random.permutation(edge_ls)
    if len(edge_ls) % 2 == 1:
        edge_ls = edge_ls[:-1]

    # edge_ls gives list of edges, with each row representing the index of the two nodes connected by the edge
    edge_ls = edge_ls.reshape(-1, 2)
    return edge_ls


def SIR_djset(n, edge_ls, lambda_):
    """Simulate the SIR model on the given network represented by disjoint set.
    Assume single seed node to be node 0."""

    C = DisjointSet(range(n))
    m = edge_ls.shape[0] # Total number of edges
    prob_infect = np.random.binomial(1, lambda_, m) # Randomly assign infection outcome to each edge based on probability lambda_
    
    for idx, edge in enumerate(edge_ls):
        if prob_infect[idx] == 1:
            C.merge(edge[0], edge[1])

    return C.subset_size(0) # Assume node 0 is the seed node


def SIR_djsets(n, edge_ls, lambdas, iter_n=1, compute_cov=False):
    """Simulate the SIR model on the given network represented by disjoint set for multiple lambda values."""
    cluster_size_means = np.zeros(len(lambdas))
    cluster_size_cov = np.zeros(len(lambdas))
    for idx, lambda_ in enumerate(lambdas):
        cluster_sizes = np.zeros(iter_n)
        for itn in range(iter_n):
            cluster_sizes[itn] = SIR_djset(n, edge_ls, lambda_)
        cluster_size_means[idx] = np.mean(cluster_sizes)
        if compute_cov:
            cluster_size_cov[idx] = np.std(cluster_sizes) / cluster_size_means[idx]
            return cluster_size_means, cluster_size_cov
    return cluster_size_means # shape (len(lambdas),)



if __name__ == "__main__":
    n = 10000
    mean = 20
    lambda_ary = np.linspace(0, 0.15, 16) 
    iter_n = 100
    mu = np.zeros(len(lambda_ary)) # Mean of the number of total infections mu over lambda
    cov = np.zeros(len(lambda_ary)) # coefficient of variation of the number of total infections sigma / mu over lambda

    # Generate one single network and calculate the number of total infections for each lambda
    edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
    # for idx, lambda_ in enumerate(lambda_ary):
    #     print("processing lambda: ", lambda_)
    #     cluster_size = np.zeros(iter_n)
    #     for itn in range(iter_n):
    #         print("iter: ", itn, '/', iter_n, end='\r')
    #         cluster_size[itn] = SIR_djset(n, edge_ls, lambda_)
    #     mu[idx] = np.mean(cluster_size)
    #     cov[idx] = np.std(cluster_size) / mu[idx]
    mu, cov = SIR_djsets(n, edge_ls, lambda_ary, iter_n, compute_cov=True)

    # # Generate a new network for each iteration, and calculate the number of total infections for each lambda
    # output = np.zeros((iter_n, len(lambda_ary))) # Store the number of total infections for each iteration and each lambda
    # for itn in range(iter_n):
    #     print("iter: ", itn, '/', iter_n)
    #     edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
    #     for lam_i, lambda_ in enumerate(lambda_ary):
    #         output[itn, lam_i] = SIR_djset(edge_ls, lambda_)

    # mu = np.mean(output, axis=0)
    # cov = np.std(output, axis=0) / mu
        
    # plt.figure()
    # plt.plot(lambda_ary, mu, "o-", label="Mean")
    # plt.plot(lambda_ary, cov, "o-", label="Coefficient of variation")
    # plt.xlabel("Transmission rate")
    # plt.ylabel("Predicted number of total infections")
    # plt.legend()
    # plt.show()


    fig, ax1 = plt.subplots()

    # Plot data on the first y-axis
    ax1.plot(lambda_ary, mu, "o-", label="Mean")
    ax1.set_xlabel('Transmission rate')
    ax1.set_ylabel("Mean")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create the second y-axis
    ax2 = ax1.twinx()
    ax2.plot(lambda_ary, cov, "o-", label="Coefficient of variation")
    ax2.set_ylabel("Coefficient of variation")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.show()



    