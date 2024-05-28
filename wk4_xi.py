import numpy as np
import matplotlib.pyplot as plt

# Theoretical calculation of dynamicl centrality from eigenvector centrality
def centrality_ary(network, lambda_):
    """Compute the dynamic centrality of each node in the network 
    from the principal eigenvector of its adjacency matrix."""
    A_prime = network.adj_m * lambda_
    # Step 3: Compute the principal eigenvector
    evalues, evectors = np.linalg.eig(A_prime)
    xi_ary = evectors[:, np.argmax(evalues.real)].real

    return xi_ary


# Empirical simulation to calculate the dynamic centrality
def centrality_ary_empirical(network, lambda_, iter_n=100):
    """Compute the dynamic centrality of each node in the network 
    from the principal eigenvector of its adjacency matrix."""
    # n = network.n
    # xi_ary = np.zeros(n)
    # for _ in range(iter_n):
    #     A_prime = network.adj_m * lambda_
    #     # Step 3: Compute the principal eigenvector
    #     evalues, evectors = np.linalg.eig(A_prime)
    #     xi_ary += evectors[:, np.argmax(evalues.real)].real

    # return xi_ary / iter_n

