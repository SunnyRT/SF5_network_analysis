import numpy as np
import matplotlib.pyplot as plt


from wk2_1_config_model import *
from wk4_1_xi import *


def friend_centrality(network, xi_ary):
    """Compute the average neighbous' dynamic centrality of a randomly chosen node in the network."""
    n = network.n
    xi_neighbour_ary = np.zeros(n)
    adj_ls = network.adj_ls
    for i in range(n):
        xi_neighbour_ary[i] = np.mean([xi_ary[j] for j in adj_ls[i]])

    # remove nan values
    xi_neighbour_ary = xi_neighbour_ary[~np.isnan(xi_neighbour_ary)]

    return xi_neighbour_ary


if __name__ == "__main__":
    n = 10000
    mean = 20

    while True:
        dist = input("Enter '1: poisson' or '2: geometric' for degree distribution ('exit'): ")

        if dist == '1':
            dist = 'poisson'
            k_ary = deg_dist_poisson(n, mean)
            lambda_crit= 0.06
        elif dist == '2':
            dist = 'geometric'
            k_ary = deg_dist_geo(n, mean)
            lambda_crit= 0.04
        elif dist == 'exit':
            break
        else:
            raise ValueError("Invalid input for degree distribution.")
        # Generate a random network
        network = config_graph_gen(n, k_ary)
        print("Network generated.")
        xi_ary = centrality_ary_si(network, lambda_crit)
        print("Centrality calculated.")
        xi_nb_ary = friend_centrality(network, xi_ary)
        print("Friend centrality calculated.")

        xi_mean = np.mean(xi_ary)
        print(f'⟨x⟩: {xi_mean:.1f}')
        xi_nb_mean = np.mean(xi_nb_ary)
        print(f'⟨Ax/k⟩: {xi_nb_mean:.1f}')

        plt.figure()
        plt.hist(xi_ary, bins=50, alpha=0.5, label=f'{dist}: node')
        plt.hist(xi_nb_ary, bins=50, alpha=0.5, label=f'{dist}: neighbour')

        # plot mean of both histograms
        
        plt.axvline(xi_mean, color='tab:blue', linestyle='dashed', linewidth=1)
        plt.text(xi_mean, 500, f'⟨x⟩: {xi_mean:.1e}', rotation=90)
        plt.axvline(xi_nb_mean, color='tab:orange', linestyle='dashed', linewidth=1)
        plt.text(xi_nb_mean, 500, f'⟨Ax/k⟩: {xi_nb_mean:.1e}', rotation=90)


        plt.xlabel("Infection probability")
        plt.ylabel("Frequency (number of nodes)")
        plt.legend()
        plt.show()

        # network_p= config_graph_gen(n, k_ary_p)
        # network_g= config_graph_gen(n, k_ary_g)
        # xi_ary_p = centrality_ary_si(network_p, lambda_crit_p)
        # xi_ary_g = centrality_ary_si(network_g, lambda_crit_g)

        # xi_nb_ary_p = friend_centrality(network_p, xi_ary_p)
        # xi_nb_ary_g = friend_centrality(network_g, xi_ary_g)

        # plt.hist(xi_ary_p, bins=50, alpha=0.5, label='Poisson node')
        # plt.hist(xi_ary_g, bins=50, alpha=0.5, label='Geometric node')

        # plt.hist(xi_nb_ary_p, bins=50, alpha=0.5, label='Poisson neighbour')
        # plt.hist(xi_nb_ary_g, bins=50, alpha=0.5, label='Geometric neighbour')