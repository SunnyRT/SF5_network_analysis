# import numpy as np
# import matplotlib.pyplot as plt
# from network_def import Network 

# def rm_graph_gen(num_nodes, p):
#     """Sample a random network G(n,p) from Bernoulli distribution with nodes n and success rate p."""
#     rm_graph = Network(adj_m=np.random.binomial(n=1, p=p, size=(num_nodes, num_nodes)))
#     # print(rm_graph.adj_m)
#     return rm_graph


# def edge_hist(num_runs, num_nodes, p_range, bins=30):
#     """Sample large number of runs of random graph G(n,p) and plot a histogram for number of edges m for each graph sample"""
#     if not isinstance(p_range, np.ndarray):
#         p_range = np.array(p_range)

#     # Initialize m_dataset for all samples and for each p value
#     m_dataset = np.empty((len(p_range), num_runs))

#     # Iterate over p_range
#     for i, p in enumerate(p_range):
#         # Generate random graphs for each p
#         rm_graphs = [rm_graph_gen(num_nodes,p) for k in range(num_runs)]
#         # Compute edge counts for each graph
#         m_ary = np.array([rm_graph.edge_count() for rm_graph in rm_graphs])
#         m_dataset[i] = m_ary
    
#     # Plot histograms for each p
#     hist_plt(m_dataset, p_range)

#     return m_dataset

  
# def hist_plt(datas, p_range, bins=30, show_plt = True):
#     """Plotting of histograms of given dataset with different p"""

#     for i,p in enumerate(p_range):
#         plt.hist(datas[i], bins=bins, alpha=0.5, label=f"p={p:.1f}")
    
#     # Add labels and title
#     plt.xlabel('Total number of edges m')
#     plt.ylabel('Frequency')
#     plt.title(f"Histograms of edges m for random graphs")
#     plt.legend()

#     if show_plt:
#         plt.show()
    

    


    
        





