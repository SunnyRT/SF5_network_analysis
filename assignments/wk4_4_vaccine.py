import numpy as np
import matplotlib.pyplot as plt


from wk2_1_config_model import *
from wk4_1_xi import *
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

def vaccine_edgels(edge_ls, v, n, vacc_friend=False):
    """Vaccinate a fraction v of the nodes in the network by removing those nodes from the network.
    vacc_friend = False: randomly selecte v*n nodes to remove.
    vacc_friend = True: randomly select v*n nodes, and each node nominate one of its friends to remove."""

    nodes = np.arange(n)
    n_remove = int(v * n)
    nodes_chosen = np.random.choice(nodes, n_remove, replace=False)

    if not vacc_friend:
        nodes_to_remove = nodes_chosen
    else:
        nodes_to_remove = sample_friends_to_remove(edge_ls, nodes_chosen, n)
    
    # Create a mask to keep edges where neither node is removed
    mask = np.isin(edge_ls, nodes_to_remove)
    mask = ~np.any(mask, axis=1)
    
    # Update the edge list by removing edges connected to the selected nodes
    updated_edge_ls = edge_ls[mask]
    
    print(updated_edge_ls.shape)
    print(f"Removed {len(nodes_to_remove)} nodes ({v*100:.2f}%) from the network.")
    return updated_edge_ls




def sample_friends_to_remove(edge_ls, nodes_chosen, n):
    """Sample the set of randomly nominated friends to vaccinate."""
    adj_ls = edge_to_adj_ls(edge_ls, n)
    friends_to_remove = []
    for i in nodes_chosen:
        # if the node no friends, skip and resample
        while len(adj_ls[i]) == 0:
            i = np.random.choice(n)
        
        friend_chosen = np.random.choice(list(adj_ls[i]))
        friends_to_remove.append(friend_chosen)

    return friends_to_remove


def edge_to_adj_ls(edge_ls, n):
    """Convert edge list to adjacency list."""
    adj_ls = [set() for _ in range(n)]
    for edge in edge_ls:
        if edge[0] != edge[1]: # ignore self-loops
            adj_ls[edge[0]].add(edge[1])
            adj_ls[edge[1]].add(edge[0])
    return adj_ls



if __name__ == '__main__':
    """ Vaccination of nodes at rate lambda = 0.0, 0.2, 0.4."""
    n = 10000
    mean = 20
    v_ary = [0.0, 0.2, 0.4]
    colorscheme = ['r', 'b', 'g']

    lam_ary = np.linspace(0, 0.3, 30)
    avg_n = 1000

        # generate one single network edge list
    edge_ls = config_graph_edge_ls(n, deg_dist_poisson(n, mean))

    # run simulations for each vaccination rate
    outputs_nodes = np.empty(len(v_ary), dtype=object)

    for idx, v in enumerate(v_ary):
        print(f"Processing vaccination rate: {v}")
        if v == 0.0: # no vaccination
            edge_ls_vi = edge_ls 
        
        else:
            edge_ls = np.random.permutation(edge_ls) # reshuffle the edge list
            edge_ls_vi = vaccine_edgels(edge_ls, v, n) # remove edges connected to vaccinated nodes

        output_nodes = SIR_ci_lambda(n, edge_ls_vi, lam_ary, avg_n, compute_mean=True) # output shape (lam_n,)
        outputs_nodes[idx] = output_nodes

        
    # outputs shape (v_n, lam_n)
    # plot the results
    plt.figure()
    for idx, v in enumerate(v_ary):
        plt.plot(lam_ary, outputs_nodes[idx], "o-", color = colorscheme[idx], label=f"v={v} (nodes)")
    plt.xlabel("Transmission rate")
    plt.ylabel("Average number of total infections")
    plt.legend()
    plt.show()




