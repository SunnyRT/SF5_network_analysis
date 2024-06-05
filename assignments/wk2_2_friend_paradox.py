import numpy as np
import matplotlib.pyplot as plt
from network1_config import *

from network_def import Network

def friend_deg_dist(graph, num_trials=10000):
    """sample the degree distribution of friends of a random node"""
    n = graph.n
    friend_degs = np.zeros(num_trials)
    for idx in range(num_trials):
        # Randomly select a node i with at least one friend
        friends_i = None
        while not friends_i:
            i = np.random.randint(n)
            friends_i = graph.neighbors(i)      
        
        # Randomly select a friend j of the node i and compute degree of j
        j = np.random.choice(list(friends_i))
        friend_degs[idx] = graph.deg(j)
        
    if 0 in friend_degs:
        raise ValueError("Zero degree found in friend degree distribution")
    return friend_degs

def i_j_deg_hist(graph):
    """Plotting histograms of degree distribution of random node i and its friends j"""
    degs = graph.deg_dist()
    friend_degs = friend_deg_dist(graph)

    # Plot histograms of degree distributions for i and j
    plt.figure()
    plt.hist(degs, bins=np.arange(np.max(degs))-0.5, density=True, alpha=0.5, label="deg of node")
    plt.hist(friend_degs, bins=np.arange(np.max(friend_degs))-0.5, density=True, alpha=0.5, label="deg of friend")
    
    # Plot mean values of degree distributions for i and j
    mean_i = np.mean(degs)
    mean_j = np.mean(friend_degs)
    plt.axvline(mean_i, color='blue', linestyle='dashed', linewidth=2, label=f'node deg mean: {mean_i:.2f}')
    plt.axvline(mean_j, color='orange', linestyle='dashed', linewidth=2, label=f'friend deg mean: {mean_j:.2f}')

    plt.xlabel('Degree')
    plt.ylabel('Probability density function (i.e. normalized frequency)')
    plt.title(f"Histograms of i, j degree distribution for a {dist} configured random graph")
    plt.legend()
    plt.show()

# wk2_3
def delta_hist(graph):
    """Plotting histograms of difference between degree of random node i and average of its friends' degrees"""
    plt.figure()
    n = graph.n
    delta_dist = []
    for i in range(n):
        friends_i = graph.neighbors(i)
        if friends_i: # only consider nodes with at least one friend
            k_i = graph.deg(i)
            kappa_i = np.mean([graph.deg(j) for j in friends_i])
            delta_i = kappa_i - k_i
            delta_dist.append(delta_i)

    bins = np.arange(np.min(delta_dist), np.max(delta_dist))-0.5
    plt.hist(delta_dist, bins=bins, density=True, alpha=0.5, label="delta distribution")
    plt.axvline(np.mean(delta_dist), color='blue', linestyle='dashed', linewidth=2, label=f'delta mean: {np.mean(delta_dist):.2f}')
    plt.xlabel('Δ (i.e. mean degree of friends - degree)')
    plt.ylabel('Probability density function (i.e. normalized frequency)')
    plt.title(f"Histograms of delta distribution for a {dist} configured random graph")
    plt.legend()
    plt.show()

# wk2_4
def friend_deg_hist_plt(graph, dist, mean):
    """Plotting of histograms of friend degree distribution of given graph
    verify that the degree distribution q_k of friends of a random node  is equal to kp_k / <k> for poisson and geo degree distribution p_k"""
    friend_degs = friend_deg_dist(graph)
    k_ary = np.arange(0, max(friend_degs)+1)

    if dist == 'poisson':
        lam = float(mean)
        p_k = np.exp(-lam) * np.power(lam, k_ary) / factorial(k_ary)
    elif dist == 'geo':
        p = 1/(1+mean)
        p_k = p*np.power(1-p, k_ary)

    q_k = k_ary * p_k # element-wise multiplication
    q_k /= np.sum(q_k) # normalize to make q_k a valid probability distribution

    plt.hist(friend_degs, bins=np.arange(np.max(friend_degs))-0.5, density=True, alpha=0.5, label="histogram of friend degree distribution")
    plt.plot(k_ary, q_k, label="expected friend degree distribution q_k")

    plt.xlabel('Friend degree')
    plt.ylabel('Probability density function (i.e. normalized frequency)')
    plt.title(f"Histograms of friend degree distribution for a {dist} configured random graph")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    n = 10000
    mean = 10

    dist = input("Enter 'poisson' or 'geo' for degree distribution: ")
    if dist == 'poisson':
            k_ary = deg_dist_poisson(n, mean)
    elif dist == 'geo':
        k_ary = deg_dist_geo(n, mean)
    else:
        raise ValueError("Invalid input for degree distribution.")
    
    # generate degree distributions for both random node i and its friends j
    graph = config_graph_gen(n, k_ary)

    while True:
        plot = input("Plot (1)i vs j deg dist; (2)delta; (3)j deg dist & q_k; (q)quit:")
        if plot == '1':
            i_j_deg_hist(graph)

        elif plot == '2':
            delta_hist(graph)

        elif plot == '3':
            friend_deg_hist_plt(graph, dist, mean)
        
        elif plot == 'q':
            break
        