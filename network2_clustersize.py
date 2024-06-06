import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import DisjointSet

from network1_config import *
from network0_def import *



def SIR_djset(n, edge_ls, lambda_):
    """ For a single, fixed lambda value, run the SIR process on a network and return the average size of the final cluster.
    Using disjoint set data structure to keep track of clusters.
    Assume single seed node to be any node and average final cluster size over all nodes."""

    C = DisjointSet(range(n))
    m = edge_ls.shape[0] # Total number of edges
    prob_infect = np.random.binomial(1, lambda_, m) # Randomly assign infection outcome to each edge based on probability lambda_
    
    for idx, edge in enumerate(edge_ls):
        if prob_infect[idx] == 1:
            C.merge(edge[0], edge[1])

    cluster_sizes = np.zeros(n)

    for subset in C.subsets():
        size = len(subset)
        subset = np.fromiter(subset, dtype=np.int32)
        cluster_sizes[subset] = size # assign the same size of the cluster to all nodes in the cluster

    mean_size = np.mean(cluster_sizes)
    return mean_size 




def SIR_djset_dir(n, edge_ls, lambda_nodes, compute_var=False, iter_n=1):
    """Simulate the SIR model on the given network.
    with asymmetric edges (i.e., directed edges) transmission."""

    # create a double edge list which counts each edge twice in both directions.
    # switch the two columns of the edge list to get the reverse edge list.
    edge_ls_reverse = np.flip(edge_ls, axis=1)
    # concatenate the two edge lists to get the double edge list.
    edge_ls_double = np.concatenate((edge_ls, edge_ls_reverse), axis=0)

    # create a Network_dir object with no edges
    network_dir = Network_dir(n)

    # assign lambda to each edge based on the sink node j=edge[1] of its lambda_j
    lambda_edges = np.array([lambda_nodes[edge[1]] for edge in edge_ls_double])
    # Randomly assign infection outcome to each edge based on lambda_edges (i.e., probability of infection of the sink node)
    # Flip the biased coin for each edge to determine if the edge is infected
    infect_edges = np.random.binomial(1, lambda_edges) 

    
    # add each directed edge to network_dir based on the infection outcome
    for idx, edge in enumerate(edge_ls_double):
        if infect_edges[idx] == 1:
            network_dir.add_edge(edge[0], edge[1])  # TODO: what if lambda is assigned to each entry of the adjacency matrix??
    
    print("Edges infected: ", network_dir.edge_count())
  
    cluster_sizes = np.zeros(iter_n)
    for i in range(iter_n):
        cluster_sizes[i] = network_dir.sink_size(i)
    
    mean_size = np.mean(cluster_sizes)
    var_size = None
    if compute_var:
        var_size = np.var(cluster_sizes)
    # return the mean cluster size of each node if being the seed node
    return mean_size, var_size




def SIR_djset_mask(n, edge_ls, lambda_, mask_states, w):
    """ For a single, fixed lambda value, run the SIR process on a network based on mask wearing 
    and return the average size of the final cluster.

    Assume single seed node to be any node and average final cluster size over all nodes."""

    w_s, w_i, w_b = w

    # create a double edge list which counts each edge twice in both directions.
    # switch the two columns of the edge list to get the reverse edge list.
    edge_ls_reverse = np.flip(edge_ls, axis=1)
    # concatenate the two edge lists to get the double edge list.
    edge_ls_double = np.concatenate((edge_ls, edge_ls_reverse), axis=0)

    # create a Network_dir object with no edges
    network_dir = Network_dir(n)



    # Assign weight value to each edge based on the mask wearing state of the source node 
    # (single direction only)
    w_edges = np.ones(edge_ls.shape[0])
    for idx, edge in enumerate(edge_ls):
        state_i = mask_states[edge[0]] # source node
        state_j = mask_states[edge[1]] # sink node

        if state_i == 0:
            if state_j == 0:
                pass
            else:
                w_edges[idx] = w_s # only susceptible sink node is wearing a mask
        else:
            if state_j == 0:
                w_edges[idx] = w_i # only infected source node is wearing a mask
            else:
                w_edges[idx] = w_b # both nodes are wearing masks

    # Account for both directions of the edge
    w_edges = np.concatenate((w_edges, swap_i_j(w_edges, w_s, w_i))) # shape (2*m,)

    # Weighted lambda value for each edge
    lambda_edges = lambda_ * w_edges

    # Randomly assign infection outcome to each edge based on lambda_edges (i.e., probability of weighted infection for each edge)
    # Flip the biased coin for each edge to determine if the edge is infected
    infect_edges = np.random.binomial(1, lambda_edges)

    # add each directed edge to network_dir based on the infection outcome
    for idx, edge in enumerate(edge_ls_double):
        if infect_edges[idx] == 1:
            network_dir.add_edge(edge[0], edge[1])
        
    print("Edges infected: ", network_dir.edge_count())

    cluster_size = network_dir.sink_size(0) # assume the seed node is node 0
    return cluster_size
    


def swap_i_j(w_edges, w_s, w_i):
    """Swap the weight values of the edges based on the source and sink node mask wearing states."""
    if w_s == w_i:
        return w_edges
    
    else:
        w_reverse = w_edges.copy()

        w_reverse[w_edges == w_s] = w_i
        w_reverse[w_edges == w_i] = w_s

    return w_reverse





# if __name__ == "__main__":
#     n = 10000
#     mean = 20
#     lambda_ary = np.linspace(0, 0.15, 16) 
#     iter_n = 100
#     mu = np.zeros(len(lambda_ary)) # Mean of the number of total infections mu over lambda
#     cov = np.zeros(len(lambda_ary)) # coefficient of variation of the number of total infections sigma / mu over lambda

#     # Generate one single network and calculate the number of total infections for each lambda
#     edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))
#     for idx, lambda_ in enumerate(lambda_ary):
#         print("processing lambda: ", lambda_)
#         cluster_size = np.zeros(iter_n)
#         for itn in range(iter_n):
#             print("iter: ", itn, '/', iter_n, end='\r')
#             cluster_size[itn] = SIR_djset(n, edge_ls, lambda_)
#         mu[idx] = np.mean(cluster_size)
#         cov[idx] = np.std(cluster_size) / mu[idx]




#     fig, ax1 = plt.subplots()

#     # Plot data on the first y-axis
#     ax1.plot(lambda_ary, mu, "o-", label="Mean")
#     ax1.set_xlabel('Transmission rate')
#     ax1.set_ylabel("Mean")
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     # Create the second y-axis
#     ax2 = ax1.twinx()
#     ax2.plot(lambda_ary, cov, "o-", label="Coefficient of variation")
#     ax2.set_ylabel("Coefficient of variation")
#     ax2.tick_params(axis='y', labelcolor='tab:orange')

#     # Add legends
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')

#     # Show the plot
#     plt.show()




# Prediction from iterative calculations of $s_i$ probabilities.
# #### TODO: Modified from wk3_2_crit_thres to account for variations in the susceptibility of nodes (different lambda_i for each node)
# def compute_s_probs(network, lambdas, tol=1e-6, max_iter=1000):
#     n = network.n
#     assert len(lambdas) == n, "Length of lambdas must equal to the number of nodes."
#     adj_ls = network.adj_ls
    
#     s = np.random.rand(n)
#     s_prev = np.zeros(n)

#     for _ in range(max_iter):
#         if np.allclose(s, s_prev, atol=tol):
#             break
#         s_prev = s.copy()

#         for i in range(n):
#             s_j = np.array([s_prev[j] for j in adj_ls[i]])
#             lambda_i = lambdas[i]
#             log_terms = np.log(1 - lambda_i + s_j * lambda_i)
#             s[i] = np.exp(np.sum(log_terms))

#     return s # shape (n,)