import numpy as np
import matplotlib.pyplot as plt
from wk2_1_config_model import *

from network_def import Network

def comp_size_avg2(n, mean, dist, num_trials=1):
    """Compute the average size of the component that contains node 1 
    for a network defined by configuration model with n nodes and degree distribution dist."""

    avg_size = 0 
    for t in range(num_trials):
        if dist == 'poisson':
            network_t = config_graph_gen(n, deg_dist_poisson(n, mean))
        elif dist == 'geo':
            network_t = config_graph_gen(n, deg_dist_geo(n, mean))
        else:
            raise ValueError("Invalid distribution type.")
        t_size = network_t.comp_size(0)
        avg_size = (t*avg_size + t_size) / (t+1)     
    return avg_size



if __name__ =='__main__':
    n = 1000
    mean_len = 11
    mean_ary = np.linspace(0, 2, mean_len)
    comp_size_poisson = np.zeros(mean_len)
    comp_size_geo = np.zeros(mean_len)
    avg_trials = 10

    for idx, mean in enumerate(mean_ary):
        print("processiong: mean = ", mean)

        comp_size_poisson[idx] = comp_size_avg2(n, mean, 'poisson', avg_trials)
        comp_size_geo[idx] = comp_size_avg2(n, mean, 'geo', avg_trials)

    plt.plot(mean_ary, comp_size_poisson/n, "-o", label="Poisson")
    plt.plot(mean_ary, comp_size_geo/n, "-o", label="Geometric")
    plt.xlabel("mean")
    plt.ylabel("mean component size (%)")
    plt.legend()
    plt.show()


