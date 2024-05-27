import numpy as np
import matplotlib.pyplot as plt

from wk2_1_config_model import *
from network_def import Network


# def SIR(network, lambda_, seed_n, Nr_only=False): 
#     """Simulate the SIR model on the given network.
#     Assuming time step is 1 week:
#         Each node is infected for exactly 1 week before recovering;
#         Infected neighbors are only infectious the next week."""
    
#     n = network.n
#     # Initialize states
#     S = np.ones(n)
#     I = np.zeros(n)
#     R = np.zeros(n)
#     seeds = np.random.choice(n, seed_n, replace=False)
#     S[seeds] = 0
#     I[seeds] = 1
    
#     Ns_t = [n-seed_n]
#     Ni_t = [seed_n]
#     Nr_t = [0]

#     while np.sum(I) > 0:
#         for i in range(n):
#                 if I[i] == 1:
#                     for j in network.neighbors(i):
#                         if S[j] == 1 and np.random.rand() < lambda_:
#                             S[j] = 0
#                             I[j] = 1
#                     I[i] = 0
#                     R[i] = 1
#         Ns_t.append(np.sum(S))
#         Ni_t.append(np.sum(I))
#         Nr_t.append(np.sum(R))
#     if Nr_only:
#         return Nr_t[-1]
#     return Ns_t, Ni_t, Nr_t

# More efficient implementation using sets
def SIR(network, lambda_, seed_n, Nr_only=False): 
    """Simulate the SIR model on the given network.
    Assuming time step is 1 week:
        Each node is infected for exactly 1 week before recovering;
        Infected neighbors are only infectious the next week."""
    n = network.n
    # Initialize states
    S = set(range(n))
    I = set()
    R = set()
    seeds = np.random.choice(n, seed_n, replace=False)
    S.difference_update(seeds)
    I.update(seeds)

    Ns_t = [n - seed_n]
    Ni_t = [seed_n]
    Nr_t = [0]

    adj_ls = network.adj_ls

    while len(I) > 0:
        new_I = set()
        new_R = set()
        for i in I:
            for j in adj_ls[i]:
                if j in S and np.random.rand() < lambda_:
                    S.remove(j)
                    new_I.add(j)
            new_R.add(i)
        I.difference_update(new_R)
        I.update(new_I)
        R.update(new_R)
        Ns_t.append(len(S))
        Ni_t.append(len(I))
        Nr_t.append(len(R))

    if Nr_only:
        return Nr_t[-1]
    return Ns_t, Ni_t, Nr_t








if __name__ == '__main__':
    # Parameters
    n = 10000
    mean = 20
    seed_n = 1

    """Plot the SIR process: number of nodes in state S, I, R over time, 
    on a network with given parameters."""

    while True:
        lambda_ = input("Enter the transmission rate('q: exit'): ")
        if lambda_ == 'q':
            break
        else: 
            lambda_ = float(lambda_)
            if lambda_ > 1 or lambda_ < 0:
                raise ValueError("Invalid transmission rate.")
        

        # Generate network
        network = config_graph_gen(n, deg_dist_poisson(n, mean))

        # Simulate SIR model
        Ns_t, Ni_t, Nr_t = SIR(network, lambda_, seed_n)

        # Plot results
        plt.plot(Ns_t, label="S")
        plt.plot(Ni_t, label="I")
        plt.plot(Nr_t, label="R")
        plt.xlabel("Time (weeks)")
        plt.ylabel("Number of nodes")
        plt.legend()
        plt.show()
    

