import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import binom
from scipy.integrate import simps

from network_def import Network 
from wk1_sampling import rm_graph_gen, rm_graph_gen2, rm_graph_gen3

def edge_hist(num_runs, num_nodes, p):
    """Sample large number of runs of random graph G(n,p) and plot a histogram for number of edges m for each graph sample"""

    # Generate random graphs 
    rm_graphs = [rm_graph_gen(num_nodes,p) for _ in range(num_runs)]
    # Compute edge counts for each graph
    m_ary = np.array([rm_graph.edge_count() for rm_graph in rm_graphs])

    # Plot histograms for each p
    edge_hist_plt(m_ary, p)

    return m_ary

  
def edge_hist_plt(data, p, show_plt = True):
    """Plotting of histograms of given p and data of m"""

    plt.hist(data, bins=np.arange(np.max(data))-0.5, density=True, alpha=0.5, label=f"p={p:.1f}")
    
    # Add labels and title
    plt.xlabel('Total number of edges m')
    plt.ylabel('Frequency')
    plt.title(f"Histograms of edges m for random graphs")
    plt.legend()

    if show_plt:
        plt.show()
    

def edge_hist_bino_plt(data, num_nodes, p):

    edge_hist_plt(data, p, show_plt = False)


    # Total possible paris of nodes
    num_2nodes = comb(num_nodes,2)

    # define binomial distribution of m
    m_mean = num_2nodes * p
    m_range = np.linspace(m_mean-150, m_mean+150, 301)

    # Pm = np.array([comb(num_2nodes, m) * (p**m) * ((1-p)**(num_2nodes-m)) for m in m_range])
    Pm = binom.pmf(m_range, num_2nodes, p)

    plt.plot(m_range, Pm, label="binomial distribution")
    plt_mean_var(m_range, Pm)
    plt.legend()
    plt.show()


def mean_var_calc(x_range, pmf):
    mean = simps(x_range * pmf, x_range)
    var = simps((x_range - mean)**2 * pmf, x_range)

    return mean, var

def plt_mean_var(x_range, pmf):
    mean, var = mean_var_calc(x_range, pmf)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
    
    std = np.sqrt(var)
    plt.axvline(mean+std, color='g', linestyle='dashed', linewidth=1)
    
    plt.axvline(mean-std, color='g', linestyle='dashed', linewidth=1)

    plt.text(mean+std, 0.001, f"mean = {mean:.2f}", color='r', ha = 'right')
    plt.text(mean+std+10, 0.005, f"var = {var:.2f}", color='g', ha = 'left')

