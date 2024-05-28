import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


from network_def import Network
from wk2_1_config_model import *
from wk3_1_SIR import *



# Updated version such that each sampling takes place on one particular graph (sufficient:))
def SIR_Nr_lam(network, seed_n, lambda_ary, iter_n=10):
    """Plot the average number of recovered nodes (i.e. Total infection counts) 
    at the end of the SIR process against the transmission rate lambda."""


    Nr_ary = np.zeros(len(lambda_ary))

    for idx, lambda_ in enumerate(lambda_ary):
        print("processing lambda: ", lambda_)
        Nr_iter_sum = 0
        for _ in range(iter_n):
            print("iter: ", _, '/', iter_n, end='\r')
            Nr_iter_sum += SIR(network, lambda_, seed_n, Nr_only=True)

        # Average number of total infections
        Nr_ary[idx] = Nr_iter_sum / iter_n

    return Nr_ary

def compute_s_prob(network, lambda_, tol=1e-6, max_iter=1000):
    n = network.n
    adj_ls = network.adj_ls
    
    s = np.random.rand(n)
    s_prev = np.zeros(n)

    for _ in range(max_iter):
        if np.allclose(s, s_prev, atol=tol):
            break
        s_prev = s.copy()

        for i in range(n):
            s_j = np.array([s_prev[j] for j in adj_ls[i]])
            log_terms = np.log(1 - lambda_ + s_j * lambda_)
            s[i] = np.exp(np.sum(log_terms))

    return s

def crit_lambda(network):
    A = network.adj_m
    A = sp.csr_matrix(A, dtype=float)
    # compute max eigenvalue of adjacency matrix A
    v_max = eigsh(A, k=1, which='LM', return_eigenvectors=False)[0] # "k=1" specifies largest eigenvalue, "LM" means "Largest Magnitude
    # compute the critical lambda
    lambda_crit = 1/v_max
    return lambda_crit



if __name__=='__main__':
    # Parameters
    n = 10000
    mean = 20
    seed_n = 1

    lambda_ary = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8])
    
    
    # Empirical simulation
    network_p = config_graph_gen(n, deg_dist_poisson(n, mean))
    network_g = config_graph_gen(n, deg_dist_geo(n, mean))
    Nr_ary_p = SIR_Nr_lam(network_p, seed_n, lambda_ary, iter_n=100)
    Nr_ary_g = SIR_Nr_lam(network_g, seed_n, lambda_ary, iter_n=100)
    print(f"exp_p: {Nr_ary_p}")
    print(f"exp_g: {Nr_ary_g}")
    
    # Theoretical prediction
    s_ary_p = np.zeros(len(lambda_ary))
    s_ary_g = np.zeros(len(lambda_ary))
    for idx, lambda_ in enumerate(lambda_ary):

        s_p = compute_s_prob(network_p, lambda_)
        s_ary_p[idx] = np.average(s_p)
        print(f"Poisson: lambda = {lambda_}, s = {np.average(s_p)}")

        s_g = compute_s_prob(network_g, lambda_)
        s_ary_g[idx] = np.average(s_g)
        print(f"Geometric: lambda = {lambda_}, s = {np.average(s_g)}")

    Nr_ary_ps = n * (1 - s_ary_p)
    Nr_ary_gs = n * (1 - s_ary_g)
    print(f"pred_p:{Nr_ary_ps}")
    print(f"pred_g:{Nr_ary_gs}")

    # Compute critical lambda
    lambda_crit_p = crit_lambda(network_p)
    lambda_crit_g = crit_lambda(network_g)
    print(f"crit_p: {lambda_crit_p}"
          f"crit_g: {lambda_crit_g}")


    plt.figure()
    plt.plot(lambda_ary, Nr_ary_p, "o-", color = 'tab:blue', label="Poisson")
    plt.plot(lambda_ary, Nr_ary_g, "o-", color = 'tab:orange', label="Geometric")
    plt.plot(lambda_ary, Nr_ary_ps, linestyle = "--", color = 'tab:blue', label="Poisson prediction")
    plt.plot(lambda_ary, Nr_ary_gs, linestyle = "--", color = 'tab:orange', label="Geometric prediction")
    plt.axvline(x=lambda_crit_p, color='r')
    plt.text(lambda_crit_p, 6000, f'λcrit = {lambda_crit_p:.3f}', rotation=90) 
                                             
    plt.axvline(x=lambda_crit_g, color='r')
    plt.text(lambda_crit_g, 6000, f'λcrit = {lambda_crit_g:.3f}', rotation=90)

    plt.xlabel("Transmission rate lambda")
    plt.ylabel("Number of total infections")
    plt.legend()
    plt.show()
