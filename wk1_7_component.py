import numpy as np


from network_def import Network
from wk1_5_eff_sampling import *

def comp_size(network):
    """Compute the size of the component that 
    contains node 1 for a given network object"""

    comp_set = set()
    def edge_follow(node):
        for neighbour in network.neighbors(node):
            if neighbour not in comp_set:
                comp_set.add(neighbour)
                # print(comp_set)
                edge_follow(neighbour)
                

    edge_follow(0)
    return len(comp_set)


def comp_size_avg(num_nodes, p, num_trials=1):
    """Compute the average size of the component that 
    contains node 1 for a random network G(n, p)"""

    mean_size = 0 
    for i in range(num_trials):
        network_i = rm_graph_gen2(num_nodes, p)
        i_size = comp_size(network_i)
        mean_size = (i*mean_size + i_size) / (i+1)     
    
    return mean_size


if __name__ =='__main__':
    p_range = np.linspace(0,0.0004, 20)
    mean_sizes = np.empty(len(p_range), dtype=float)
    for i, p in enumerate(p_range):
        mean_sizes[i] = comp_size_avg(4096, p, num_trials=20)

    mean_sizes_ratio = mean_sizes / 4096

    plt.plot(p_range, mean_sizes_ratio, "-o")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.axvline(x=1/4095, color='r', linestyle='--')
    plt.text(1/4095, 0, "p=1/(n-1)", rotation=90, va="bottom", ha="right")
    plt.xlabel("p")
    plt.ylabel("mean component size (%)")
    plt.show()


        

