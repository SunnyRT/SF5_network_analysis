import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from network_def import Network 

def rm_graph_gen(n, p, m_only=False):
    """Naive sample a random network G(n,p) from Bernoulli distribution with nodes n and success rate p.
    Time complexity: O(n^2)"""

    adj_m = np.zeros((n, n), dtype=int)
    for i in np.ndindex(n, n):
        if i[0] < i[1]:
            adj_m[i] = np.random.binomial(1, p)
            
            # To copy the adjacency information from one direction to its opposite direction
            adj_m[i[::-1]] = adj_m[i] # [::-1] is a slicing operatiion that resverses the order of elements
    if m_only:
        return adj_m
    else:
        rm_graph = Network(adj_m=adj_m)
        # print(rm_graph.adj_m)
        return rm_graph

def rm_graph_gen2(n, p, m_only=False):
    """2-stage graph genertion sampling based on binomial edge distribution m
    Time complexity: O(n)"""

    nC2 = comb(n, 2)
    m = np.random.binomial(nC2, p)

    adj_m = np.zeros((n, n), dtype=int)

    m_idx = 0
    while m_idx < m:
        # Randomly select a pair of nodes (i,j)
        # MISTAKE: This is not a uniform sampling!!!
        # i = np.random.randint(0, n-1)
        # j = np.random.randint(i+1, n)

        # u, v = np.random.choice(n, 2, replace=False)
        # i = min(u,v)
        # j = max(u,v)

        # Correction: Uniform sampling of a pair of nodes (i,j) from lecture        
        u = np.random.randint(n-1) + 1
        v = np.random.randint(n)
        j = max(u,v)
        i = np.random.randint(j)
        # Connect the nodes (i,j) if not already connected
        if adj_m[i][j] == 0:
            adj_m[i][j] = 1
            adj_m[j][i] = 1
            m_idx +=1
    
    if m_only:
        return adj_m
    else:
        rm_graph = Network(adj_m=adj_m)
        # print(rm_graph.adj_m)
        return rm_graph


def rm_graph_gen3(n, p, m_only=False):
    """With built-in sampling library: np.random.binomial().
    Sample a random network G(n,p) from Bernoulli distribution with nodes n and success rate p.
    Time complexity: O(n^2)"""

    adj_m_upper = np.random.binomial(1,p, size = (n, n))
    adj_m = np.triu(adj_m_upper) + np.triu(adj_m_upper,1).T
    # Remove self-connection edges
    np.fill_diagonal(adj_m, 0) 

    if m_only:
        return adj_m
    else:
        rm_graph = Network(adj_m=adj_m)
        # print(rm_graph.adj_m)
        return rm_graph
    



    

    


    
        





