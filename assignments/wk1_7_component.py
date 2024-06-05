import numpy as np
from matplotlib import pyplot as plt

from network_def import Network
from wk1_sampling import rm_graph_gen2


def comp_size_avg(n, p, num_trials=1):
    """Compute the average size of the component that 
    contains node 1 for a random network G(n, p)"""

    avg_size = 0 
    for i in range(num_trials):
        network_i = rm_graph_gen2(n, p)
        i_size = network_i.comp_size(0)
        avg_size = (i*avg_size + i_size) / (i+1)     
    
    return avg_size



if __name__ =='__main__':
    p_range = np.linspace(0,0.001, 50)
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


        












# Functions to compute component of individual sets have been integrated into Network class methods
# ****************************************************************
# # Self attempt 
# # Time complexity is similar, but may be constrained by python recursion depth limit
# def comp_size(network):
#     """Compute the size of the component that 
#     contains node 1 for a given network object"""

#     c = set()
#     def edge_follow(i): # i represents node i
#         for j in network.neighbors(i): # j represents neighbour node j of node i
#             if j not in c:
#                 c.add(j)
#                 # print(c)
#                 edge_follow(j)
                
#     edge_follow(0)
#     return len(c)

# # Solution provided
# def find_component(G, i):
#     """Find the component that contains node i for a given network object"""
#     c = set()
#     q = [i]
#     while len(q) > 0:
#         j = q.pop()
#         c.add(j)
#         q += G.neighbors(j) - c  # python type overloading
#     return c