import numpy as np

# Define network object with implementation of numpy arrays
class Network(object):
    def __init__(self, num_nodes=None, adj_m=None):
        # 2 constructor through either number of nodes or adjacency matrix
        # Attributes: 
            # number of nodes <num_nodes>; 
            # adjacency list <adj_ls>; 
            # adjacency matrix <adj_m>.
         
        if adj_m is not None:
            if adj_m.shape[0] != adj_m.shape[1]:
                raise ValueError("Adjacency matrix must be square matrix.")
            if num_nodes is not None and num_nodes != adj_m.shape[0]:
                raise ValueError("Inconsistent input for graph definition.")
            self.adj_m = adj_m
            self.num_nodes = adj_m.shape[0]
            self.adj_ls = np.array([set(np.nonzero(self.adj_m[i])[0]) for i in range(self.num_nodes)])
        
        elif num_nodes is not None:
            self.num_nodes = num_nodes
            self.adj_ls = np.empty(num_nodes, dtype=object)
            self.adj_ls.fill(set())
            self.adj_m = np.zeros((num_nodes,num_nodes), dtype = bool)
        
        else:
            raise ValueError("Missing argument for graph definition.")


    def add_edge(self, i, j):
        self.adj_ls[i].append(j)
        self.adj_ls[j].append(i)
        self.adj_m[i][j] = 1
        self.adj_m[j][i] = 1

    def neighbors(self, i):
        return self.adj[i]
    
    def edge_list(self):
        return [(i,j) for i in self.adj for j in self.adj[i] if i<j]
    
    def edge_count(self):
        return np.count_nonzero(self.adj_m)
    

# # Define network object with adjacency lists
# class Network(object):
#     def __init__(self, num_nodes):
#         self.adj = {i:set() for i in range(num_nodes)} # dictionary whcih map each node to a set of connected adjacent nodes

#     def add_edge(self, i, j):
#         self.adj[i].add(j)
#         self.adj[i].add(i)
        
#     def neighbors(self, i):
#         return self.adj[i]
    
#     def edge_list(self):
#         return [(i,j) for i in self.adj for j in self.adj[i] if i<j]



# # Alternative definition with adjency matrix (np array)
# class Network_mat(object):
#     def __init__(self, num_nodes):
#         self.num_nodes = num_nodes
#         self.adj = np.zeros((num_nodes,num_nodes), dtype = bool)

#     def add_edge(self, i, j):
#         self.adj[i][j] = 1
#         self.adj[j][i] = 1

#     def neighbors(self, i):
#         # return [j for j in range(self.num_nodes) if self.adj[i][j]==1]
#         return np.nonzero(self.adj[i])[0]
    
#     def edge_list(self):
#         return [(i,j) for i in range(self.num_nodes) for j in self.neighbors(i) if i<j]

