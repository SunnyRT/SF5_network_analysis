import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import binom
from scipy.integrate import simps

from network_def import Network 
from wk1_1_naive_sampling import hist_plt


def hist_bino_plt(data, num_nodes, p):

    hist_plt(data, p, bins=30, show_plt = False)


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

