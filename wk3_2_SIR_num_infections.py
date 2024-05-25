import numpy as np
import matplotlib.pyplot as plt


from network_def import Network
from wk2_1_config_model import *
from wk3_1_SIR_process import *




def SIR_Nr_lam(n, mean, seed_n, lambda_ary, iter_n=10, k_dist=deg_dist_poisson):
    """Plot the average number of recovered nodes (i.e. Total infection counts) 
    at the end of the SIR process against the transmission rate lambda."""
    
    # """Also comparison with the theoretical number of recovered nodes 
    # based on average non-infected probability s for each lambda value."""

    Nr_ary = np.zeros(len(lambda_ary))
    # s_ary = np.zeros(len(lambda_ary))

    for idx, lambda_ in enumerate(lambda_ary):
        print("processing lambda: ", lambda_)
        Nr_iter_sum = 0
        # s_computed = False
        for _ in range(iter_n):
            print("iter: ", _, '/', iter_n, end='\r')
            network = config_graph_gen(n, k_dist(n, mean))
            Nr_iter_sum += SIR(network, lambda_, seed_n, Nr_only=True)

            # # Compute s only once for each lambda value
            # if not s_computed:
            #     s_ary[idx] = np.average(compute_s_prob(network, lambda_)[1])
            #     s_computed = True

        # Average number of total infections
        Nr_ary[idx] = Nr_iter_sum / iter_n

    # # Theoretical number of total infections based on average non-infected probability s
    # Nr_ary_s = n * (1 - s_ary)   

    return Nr_ary


# def compute_s_prob(network, lambda_, tol=1e-6, max_iter=1000):
#     n = network.n
#     adj_ls = network.adj_ls
    
#     s = np.random.rand(n)
#     s_prev = np.zeros(n)

#     for _ in range(max_iter):
#         if np.allclose(s, s_prev, atol=tol):
#             break
#         s_prev = s.copy()

#         for i in range(n):
#             s_j = np.array([s_prev[j] for j in adj_ls[i]])
#             log_terms = np.log(1 - lambda_ + s_j * lambda_)
#             s[i] = np.exp(np.sum(log_terms))

#     return s_prev, s


if __name__=='__main__':
    # Parameters
    n = 10000
    mean = 20
    seed_n = 1

    lambda_ary_p = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.5])
    lambda_ary_g = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    Nr_ary_poisson = SIR_Nr_lam(n, mean, seed_n, lambda_ary_p, iter_n=20, k_dist=deg_dist_poisson)
    Nr_ary_geo = SIR_Nr_lam(n, mean, seed_n, lambda_ary_g, iter_n=20, k_dist=deg_dist_geo)
    print(Nr_ary_poisson)
    print(Nr_ary_geo)
    
    
    
    
    
    
    
    
    
    
    
    plt.scatter(lambda_ary_p, Nr_ary_poisson, "o-", color = 'tab:blue', label="Poisson")
    plt.scatter(lambda_ary_g, Nr_ary_geo, "o-", color = 'tab:orange', label="Geometric")
    plt.xlabel("Transmission rate lambda")
    plt.ylabel("Average number of total infections")

    plt.legend()
    plt.show()
