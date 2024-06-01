import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import truncnorm

from wk2_1_config_model import config_graph_gen, deg_dist_poisson
from wk3_4_SIR_djset import SIR_djset
from wk3_2_crit_thres import compute_s_prob


def node_lambda(n, mean, var, dist="normal"):
    std = var**0.5
    """Generate the susceptibility for each node in the network."""
    if dist == "normal":
        a, b = (0 - mean) / std, (1 - mean) / std # lower and upper bounds in terms of std
        lambda_i = truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
    elif dist == "uniform":
        a = mean - std*(3**0.5)
        b = mean + std*(3**0.5) 
        lambda_i = np.random.uniform(a, b, n)
    elif dist == "exponential":
        lambda_i = np.random.exponential(mean, n)
    else:
        raise ValueError("Invalid distribution type.")
    return lambda_i

def lambda_mat(n, means, var, dist="normal"):
    """Generate the susceptibility matrix for each node in the network, for each value of lambda mean."""
    lambda_mat = np.empty((len(means),n))
    for idx, mean in enumerate(means):
        lambda_i = node_lambda(n, mean, var, dist)
        lambda_mat[idx] = lambda_i

    return lambda_mat # shape (len(means), n)


#### TODO: Modified from wk3_2_crit_thres to account for variations in the susceptibility of nodes (different lambda_i for each node)
def compute_s_probs(network, lambdas, tol=1e-6, max_iter=1000):
    n = network.n
    if type(lambdas) == float:
        lambdas = np.full(n, lambdas) # convert to array
    else:
        assert len(lambdas) == n, "Length of lambdas must equal to the number of nodes."
    adj_ls = network.adj_ls
    
    s = np.random.rand(n)
    s_prev = np.zeros(n)

    for _ in range(max_iter):
        if np.allclose(s, s_prev, atol=tol):
            break
        s_prev = s.copy()

        for i in range(n):
            s_j = np.array([s_prev[j] for j in adj_ls[i]])
            lambda_i = lambdas[i]
            log_terms = np.log(1 - lambda_i + s_j * lambda_i)
            s[i] = np.exp(np.sum(log_terms))

    return s # shape (n,)



#### TODO: Modified from wk3_4_SIR_djset to account for variations in the susceptibility of nodes (different lambda_i for each node) --> directional edges
def SIR_djset_dir(n, edge_ls, lambdas):
    """Simulate the SIR model on the given network represented by disjoint set.
    Assume single seed node to be node 0."""

    C = DisjointSet(range(n))
    m = edge_ls.shape[0] # Total number of edges
    edge_lambdas = np.array([lambdas[edge[0]] for edge in edge_ls]) # FIXME: (Simplifications) Assume the first node in the edge is the sink node (i.e. the node that can be infected)
    prob_infect = np.random.binomial(1, edge_lambdas) # Randomly assign infection outcome to each edge based on probability lambda_i (i.e., susceptibility of the sink node)
    
    for idx, edge in enumerate(edge_ls):
        if prob_infect[idx] == 1:
            C.merge(edge[0], edge[1])

    return C.subset_size(0) # Assume node 0 is the seed node



#### Theoretical prediction using the si probability (assume indepence between infection events)
def wk5_1_theoretical_comp(n, network, lambda_means, lambda_mat):
    s_ary= np.zeros(len(lambda_means))
    for idx, lambda_mean in enumerate(lambda_means):
        lambdas = lambda_mat[idx] # for a particular lambda_mean, get the corresponding lambda_i for all nodes in the network
        s_i = compute_s_probs(network, lambdas)
        s_ary[idx] = np.average(s_i) # average over all nodes
        print(f"Poisson: lambda_mean = {lambda_mean}, s = {s_ary[idx]}")

    Nr_ary_theo = n * (1 - s_ary)
    
    return Nr_ary_theo


#### Empirical simulation
def wk5_1_empirical_comp(n, edge_ls, lambda_means):
    Nr_ary_emp = np.zeros(len(lambda_means))
    Nr_cov_emp = np.zeros(len(lambda_means))
    iter_n = 100

    for idx, lambdas in enumerate(lambda_mat):
        print("processing lambda_mean: ", lambda_means[idx])
        cluster_size = np.zeros(iter_n)
        for itn in range(iter_n):
            print("iter: ", itn, '/', iter_n, end='\r')
            cluster_size[itn] = SIR_djset_dir(n, edge_ls, lambdas)
        Nr_ary_emp[idx] = np.mean(cluster_size)
        Nr_cov_emp[idx] = np.std(cluster_size) / Nr_cov_emp[idx]
    
    return Nr_ary_emp, Nr_cov_emp

############################################################################################



def main():
    n = 10000
    mean = 20
    dist = "normal"
    lambda_means = np.linspace(0, 0.3, 30)
    # lambda_vars = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    lambda_var = 0.1
    network = config_graph_gen(n, deg_dist_poisson(n, mean))
    edge_ls = network.edge_list()
    lambda_mat = lambda_mat(n, lambda_means, lambda_var, dist) # shape (len(lambda_means), n)

    # Theoretical prediction
    Nr_ary_theo = wk5_1_theoretical_comp(n, network, lambda_means, lambda_mat)
    
    # Empirical simulation
    Nr_ary_emp, Nr_cov_emp = wk5_1_empirical_comp(n, edge_ls, lambda_means)


    
    plt.plot(Nr_ary_theo, label="Theoretical")
    plt.plot(Nr_ary_emp, label="Empirical")
    plt.xlabel("Lambda")
    plt.ylabel("Total Infections")
    plt.legend()
    plt.show()