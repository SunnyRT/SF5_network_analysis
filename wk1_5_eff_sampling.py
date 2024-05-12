import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import binom
from network_def import Network 

def rm_graph_gen2(num_nodes, p, m_only=False):
    """2-stage graph genertion sampling based on binomial edge distribution m"""

    num_nodes2 = comb(num_nodes, 2)
    m = np.random.binomial(num_nodes2, p)

    adj_m = np.zeros((num_nodes, num_nodes), dtype=int)
    
    # Randomly select m distinct upper triangle indices without diagonal
    triu_id = np.triu_indices(num_nodes, k=1)
    edge_id = np.random.choice(len(triu_id[0]), m, replace=False)

    # Set the values at selected indices to 1
    adj_m[triu_id[0][edge_id], triu_id[1][edge_id]] = 1

    # Mirror the upper triangle indices with lower triangle indices to obtain the symmertric adjacency matrix
    adj_m = adj_m + adj_m.T

    
    if m_only:
        return adj_m
    else:
        rm_graph = Network(adj_m=adj_m)
        # print(rm_graph.adj_m)
        return rm_graph



def rm_graph_gen3(num_nodes, p, m_only=False):
    """With built-in sampling library: np.random.binomial().
    Sample a random network G(n,p) from Bernoulli distribution with nodes n and success rate p."""

    adj_m_upper = np.random.binomial(1,p, size = (num_nodes, num_nodes))
    adj_m = np.triu(adj_m_upper) + np.triu(adj_m_upper,1).T
    # Remove self-connection edges
    np.fill_diagonal(adj_m, 0) 

    if m_only:
        return adj_m
    else:
        rm_graph = Network(adj_m=adj_m)
        # print(rm_graph.adj_m)
        return rm_graph