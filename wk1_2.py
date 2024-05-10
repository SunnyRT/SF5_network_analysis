import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import binom

from network_def import Network 
from wk1_1 import hist_plt


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
    plt.legend()
    plt.show()
