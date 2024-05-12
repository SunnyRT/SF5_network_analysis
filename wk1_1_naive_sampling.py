import numpy as np
import matplotlib.pyplot as plt
from network_def import Network 

def rm_graph_gen(num_nodes, p, m_only=False):
    """Naive sample a random network G(n,p) from Bernoulli distribution with nodes n and success rate p."""
    adj_m = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in np.ndindex(num_nodes, num_nodes):
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


    
def edge_hist(num_runs, num_nodes, p, bins=30):
    """Sample large number of runs of random graph G(n,p) and plot a histogram for number of edges m for each graph sample"""

    # Generate random graphs 
    rm_graphs = [rm_graph_gen(num_nodes,p) for k in range(num_runs)]
    # Compute edge counts for each graph
    m_ary = np.array([rm_graph.edge_count() for rm_graph in rm_graphs])

    # Plot histograms for each p
    hist_plt(m_ary, p)

    return m_ary

  
def hist_plt(data, p, bins=30, show_plt = True):
    """Plotting of histograms of given p and data of m"""

    plt.hist(data, bins=bins, density=True, alpha=0.5, label=f"p={p:.1f}")
    
    # Add labels and title
    plt.xlabel('Total number of edges m')
    plt.ylabel('Frequency')
    plt.title(f"Histograms of edges m for random graphs")
    plt.legend()

    if show_plt:
        plt.show()
    

    


    
        





