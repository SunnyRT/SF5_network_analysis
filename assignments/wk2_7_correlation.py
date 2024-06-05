import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from wk2_1_config_model import *
from network_def import Network

def correl_graph_gen(n, dist, mean):
    """Generate a correlated graph from a random configuration graph by edge rewiring."""
    if dist == 'poisson':
        k_ary = deg_dist_poisson(n, mean)
    elif dist == 'geo':
        k_ary = deg_dist_geo(n, mean)
    else:
        raise ValueError("Invalid distribution type.")
    
    adj_m = config_graph_gen(n, k_ary, m_only=True)
    G = nx.from_numpy_array(adj_m)
    G = assortative_rewiring(G)
    return G


def assortative_rewiring(G, target_assortativity=0.1, max_itn=500):
    print("Initial assortativity: ", nx.degree_assortativity_coefficient(G))
    itn = 0
    while nx.degree_assortativity_coefficient(G) < target_assortativity and itn < max_itn:
        print("Assortativity: ", nx.degree_assortativity_coefficient(G))
        # Randomly pick two edges in the graph
        (u, v), (x, y) = random.sample(G.edges(), 2)
        
        # Ensure the selected edges are unique and the nodes are distinct
        if len({u, v, x, y}) == 4:
            # Calculate current assortativity
            current_assortativity = nx.degree_assortativity_coefficient(G)
            
            # Rewire the edges
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(u, x)
            G.add_edge(v, y)
            
            # Calculate new assortativity
            new_assortativity = nx.degree_assortativity_coefficient(G)
            
            # If the new assortativity is not better, revert the change
            if new_assortativity < current_assortativity:
                G.remove_edge(u, x)
                G.remove_edge(v, y)
                G.add_edge(u, v)
                G.add_edge(x, y)
        itn += 1
    return G
        


    
if __name__ == '__main__':
    # Parameters
    n = 10000
    mean = 10

    while True:
        dist = input("Enter 'poisson' or 'geo' for degree distribution ('exit'): ")
        
        G = correl_graph_gen(n, dist, mean)
        degs = [d for n, d in G.degree()]
        plt.hist(degs, bins=np.arange(np.max(degs))-0.5, density=True, alpha=0.5, label="Graph degree distribution")
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()
