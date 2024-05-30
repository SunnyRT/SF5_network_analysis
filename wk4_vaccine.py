import numpy as np
import matplotlib.pyplot as plt

from wk2_1_config_model import *
from wk4_xi import *
from wk3_4_SIR_djset import config_graph_edge_ls
from wk3_5_SIR_cik import SIR_ci_lambda

# denote vaccination rate as v


# def vaccine_adjm(adj_m, v):
#     """Vaccinate a fraction v of the nodes in the network by removing those nodes from the network."""
#     n = adj_m.shape[0]
#     v_n = int(n * v)
#     vaccinated_nodes = np.random.choice(n, v_n, replace=False)
    
#     # create a mask to keep nodes that are not vaccinated
#     mask = np.ones(n, dtype=bool)
#     mask[vaccinated_nodes] = False

#     # remove the vaccinated nodes from the adjacency matrix
#     adj_m = adj_m[mask][:, mask]
#     print(adj_m.shape)

#     return adj_m

def vaccine_edgels(edge_ls, v, nodes_to_remove=None):
    """Vaccinate a fraction v of the nodes in the network by removing those nodes from the network."""
    # Get the unique nodes from the edge list
    nodes = np.unique(edge_ls)
    num_nodes = len(nodes)
    num_remove = int(v * num_nodes)
    
    if nodes_to_remove is None:
        # Randomly select nodes to remove
        nodes_to_remove = np.random.choice(nodes, num_remove, replace=False)
    
    # Create a mask to keep edges where neither node is removed
    mask = np.isin(edge_ls, nodes_to_remove)
    mask = ~np.any(mask, axis=1)
    
    # Update the edge list by removing edges connected to the selected nodes
    updated_edge_ls = edge_ls[mask]
    
    print(updated_edge_ls.shape)
    print(f"Removed {num_remove} nodes ({v*100:.2f}%) from the network.")
    return updated_edge_ls

def edge_to_adj_ls(edge_ls, n):
    """Convert edge list to adjacency list."""
    adj_ls = [set() for _ in range(n)]
    for edge in edge_ls:
        if edge[0] != edge[1]:
            adj_ls[edge[0]].add(edge[1])
            adj_ls[edge[1]].add(edge[0])
    return adj_ls

def sample_nb(v, n):
    """Sample the number of nodes to vaccinate."""
    return None

if __name__=="__main__":
        
    n = 10000
    mean = 20
    v_ary = [0.0, 0.2, 0.4]
    lam_ary = np.linspace(0, 0.3, 30)
    avg_n = 100


    # generate one single network edge list
    edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))

    # run simulations for each vaccination rate
    outputs = np.empty(len(v_ary), dtype=object)
    for idx, v in enumerate(v_ary):
        print(f"Processing vaccination rate: {v}")
        if v == 0:
            edge_ls_v = edge_ls # no vaccination
        else:
            edge_ls = np.random.permutation(edge_ls) # reshuffle the edge list
            edge_ls_v = vaccine_edgels(edge_ls, v) # remove edges connected to vaccinated nodes

        output = SIR_ci_lambda(n, edge_ls_v, lam_ary, avg_n, compute_mean=True) # output shape (lam_n,)
        outputs[idx] = output
        
    # outputs shape (v_n, lam_n)
    # plot the results
    plt.figure()
    for idx, v in enumerate(v_ary):
        plt.plot(lam_ary, outputs[idx], "o-", label=f"Vaccination rate: {v}")
    plt.xlabel("Transmission rate")
    plt.ylabel("Average number of total infections")
    plt.legend()
    plt.show()
