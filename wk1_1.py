import numpy as np
import matplotlib.pyplot as plt
from network_def import Network 

def rm_graph_gen(num_nodes, p):
    """Sample a random network G(n,p) from Bernoulli distribution with nodes n and success rate p."""
    rm_graph = Network(adj_m=np.random.binomial(n=1, p=p, size=(num_nodes, num_nodes)))
    # print(rm_graph.adj_m)
    return rm_graph


def edge_hist(num_runs, num_nodes, p_range, bins=30):
    """Sample large number of runs of random graph G(n,p) and plot a histogram for number of edges m for each graph sample"""
    if not isinstance(p_range, np.ndarray):
        p_range = np.array(p_range)

    m_dataset = np.empty(0)
    for p in p_range:
        m_ary = np.empty(0)
        for k in range(num_runs):
            rm_graph = rm_graph_gen(num_nodes,p)
            m = rm_graph.edge_count()
            m_ary = np.append(m_ary, m)

        # Plotting of histogram for each set of data with different p
        plt.hist(m_ary, bins=bins, alpha=0.5, label=f"p={p:.1f}")
    
    # Add labels and title
    plt.xlabel('Total number of edges m')
    plt.ylabel('Frequency')
    plt.title(f"Histograms of edges m for random graphs")
    plt.ylim(0, num_runs/5)
    plt.legend()
    plt.show()


        
    


# def hist_plt(data, p=None, bins=30):
#     """Plotting of histograms of given dataset"""

#     plt.hist(data, bins=bins)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of number of edges m, p={p:.2f}')
#     plt.show()


    


    
        





