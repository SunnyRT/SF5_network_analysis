import numpy as np
import matplotlib.pyplot as plt

from wk2_1_config_model import *
from wk4_xi import *

# denote vaccination rate as v


n = 10000
mean = 20
v_ary = [0.0, 0.2, 0.4]
lam_ary = np.linspace(0, 0.8, 17)



def vaccine_adjm(adj_m, v):
    """Vaccinate a fraction v of the nodes in the network by removing those nodes from the network."""
    n = adj_m.shape[0]
    v_n = int(n * v)
    vaccinated_nodes = np.random.choice(n, v_n, replace=False)
    
    # create a mask to keep nodes that are not vaccinated
    mask = np.ones(n, dtype=bool)
    mask[vaccinated_nodes] = False

    # remove the vaccinated nodes from the adjacency matrix
    adj_m = adj_m[mask][:, mask]
    print(adj_m.shape)

    return adj_m



# n = 10000
# mean = 20
# lambda_ary = np.linspace(0.00, 0.15, 16) 
# iter_n = 10
# mu = np.zeros(len(lambda_ary)) # Mean of the number of total infections mu over lambda
# cov = np.zeros(len(lambda_ary)) # coefficient of variation of the number of total infections sigma / mu over lambda

# # Generate a new network for each iteration, and calculate the number of total infections for each lambda
# output = np.empty(iter_n, dtype=object) # Store the number of total infections for each iteration and each lambda
# # generate one single network for all iterations
# edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))


# # TODO: Run for one iteration only, take mean and cov over all nodes
# edge_ls = np.random.permutation(edge_ls) # generate an edge list
# output= SIR_ci_lambda(n, edge_ls, lambda_ary) # output shape (lam_n, n)
# print("output matrix:", output) 
# # Compute mean and coefficient of variation over all nodes
# mu = np.mean(output, axis=1)        # shape (lam_n,)
# cov = np.std(output, axis=1) / mu   # shape (lam_n,)

# print("mu:", mu)
# print("cov:", cov)
# plt.figure()
# plt.errorbar(lambda_ary, mu, yerr=cov, fmt='o')
# plt.xlabel("Transmission rate")
# plt.ylabel("Average number of total infections")
# plt.savefig("wk3_5_SIR_cik.png")
# plt.show()    
