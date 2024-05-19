import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from wk2_1_config_model import *


from network_def import Network

def friend_deg_dist(graph, num_trials=10000):
    """sample the degree distribution of friends of a random node"""
    n = graph.n
    friend_degs = np.zeros(num_trials)
    for idx in range(num_trials):
        # Randomly select a node i
        i = np.random.randint(n)
        friends_i = graph.neighbors(i)

        # If node i has no friends, skip to next iteration
        if not friends_i:
            continue
        else:
            # Randomly select a friend j of the node i and compute degree of j
            j = np.random.choice(list(friends_i))
            friend_degs[idx] = graph.deg(j)
    return friend_degs

def i_j_deg_hist(graph, num_trials=10000):
    """Plotting of histograms of degree distribution of random node i and its friends j"""
    degs = graph.deg_dist()
    friend_degs = friend_deg_dist(graph)

    # Plot histograms of degree distributions for i and j
    plt.hist(degs, bins=30, density=True, alpha=0.5, color = "blue", label="Deg dist of random node i")
    plt.hist(friend_degs, bins=30, density=True, alpha=0.5, color = "orange", label="Deg dist of friends j of random node i")
    
    # Plot mean values of degree distributions for i and j
    mean_i = np.mean(degs)
    mean_j = np.mean(friend_degs)
    plt.axvline(mean_i, color='blue', linestyle='dashed', linewidth=2, label=f'node deg dist mean: {mean_i:.2f}')
    plt.axvline(mean_j, color='orange', linestyle='dashed', linewidth=2, label=f'friend deg dist mean: {mean_j:.2f}')

    plt.xlabel('Degree')
    plt.ylabel('Probability density function (i.e. normalized frequency)')
    plt.title(f"Histograms of i, j degree distribution for a {dist} configured random graph")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    n = 10000
    mean = 10
    while True:
        dist = input("Enter 'poisson' or 'geo' for degree distribution ('exit'): ")
        if dist == 'poisson':
                k_ary = deg_dist_poisson(n, mean)
        elif dist == 'geo':
            k_ary = deg_dist_geo(n, mean)
        elif dist == 'exit':
            break
        else:
            raise ValueError("Invalid input for degree distribution.")
        
        # generate degree distributions for both random node i and its friends j
        graph = config_graph_gen(n, k_ary)
        i_j_deg_hist(graph)
        