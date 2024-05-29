import numpy as np
import matplotlib.pyplot as plt

from wk2_1_config_model import *
from wk3_2_crit_thres import compute_s_prob

# # Theoretical calculation of dynamicl centrality from eigenvector centrality
# def centrality_ary(network, lambda_):
#     """Compute the dynamic centrality of each node in the network 
#     from the principal eigenvector of its adjacency matrix."""
#     A_prime = network.adj_m * lambda_
#     # Step 3: Compute the principal eigenvector
#     evalues, evectors = np.linalg.eig(A_prime)
#     xi_ary = evectors[:, np.argmax(evalues.real)].real

#     return xi_ary / np.sum(xi_ary) # Normalize the centrality values


# Simulation to calculate the dynamic centrality (solve by iteration)
def centrality_ary_si(network, lambda_):
    """Compute the dynamic centrality of each node in the network 
    by xi = 1-si."""
    
    si_ary = compute_s_prob(network, lambda_)
    xi_ary = 1 - si_ary

    return xi_ary 




if __name__ == "__main__":
    n = 10000
    mean = 20

    # Generate a random network

    k_ary_p = deg_dist_poisson(n, mean)
    lambda_crit_p = 0.06

    k_ary_g = deg_dist_geo(n, mean)
    lambda_crit_g = 0.04



    network_p= config_graph_gen(n, k_ary_p)
    network_g= config_graph_gen(n, k_ary_g)
    xi_ary_p = centrality_ary_si(network_p, lambda_crit_p)
    xi_ary_g = centrality_ary_si(network_g, lambda_crit_g)


    # Write the output to a text file
    f = open('wk4_results.txt', mode='a')
    f.write("Poisson: \n")
    f.write(f"Poisson degree distribution: {k_ary_p}\n")
    f.write(f"Infection prob distribution: {xi_ary_p}\n")

    f.write("Geometric: \n")
    f.write(f"Geometric degree distribution: {k_ary_g}\n")
    f.write(f"Infection prob distribution: {xi_ary_g}\n")

    plt.figure()

    plt.scatter(k_ary_g, xi_ary_g, alpha=0.2, color="tab:orange", label = "Geometric")
    plt.scatter(k_ary_p, xi_ary_p, alpha=0.2, color="tab:blue", label = "Poisson")

    plt.xlabel("Structural centrality: Degree")
    plt.ylabel("Dynamic centrality: Infection probability")
    plt.legend()
    plt.show()


