import numpy as np

# Define network object with implementation of numpy arrays
class Network(object):
    def __init__(self, n=None, adj_m=None):
        # FIXME: 2 constructor through either number of nodes or adjacency matrix
        # TODO: create a function convert a matrix into a network object
        # Attributes: 
            # number of nodes <n>; 
            # adjacency list <adj_ls>; 
            # adjacency matrix <adj_m>.
 
        if adj_m is not None:
            # Check that adjacency matrix must be square, symmetric, all diagonal terms = 0
            if adj_m.shape[0] != adj_m.shape[1]:
                raise ValueError("Adjacency matrix must be square matrix.")
            elif not np.all(np.diag(adj_m) == 0):
                raise ValueError("Any node cannot have an edge with itsef.")
            elif not np.allclose(adj_m, adj_m.T):
                raise ValueError("Every edge must be reciprocal.")
            # Check adjacency matrix shape is consistent with number of nodes
            elif n is not None and n != adj_m.shape[0]:
                raise ValueError("Inconsistent input for graph definition.")
            else:
                self.adj_m = adj_m
                self.n = adj_m.shape[0]
                self.adj_ls = np.array([set(np.nonzero(self.adj_m[i])[0]) for i in range(self.n)])
        
        elif n is not None:
            self.n = n
            self.adj_ls = np.empty(n, dtype=object)
            self.adj_ls.fill(set())
            self.adj_m = np.zeros((n,n), dtype = bool)
        
        else:
            raise ValueError("Missing argument for graph definition.")


    def add_edge(self, i, j):
        self.adj_ls[i].append(j)
        self.adj_ls[j].append(i)
        self.adj_m[i][j] = 1
        self.adj_m[j][i] = 1

    def neighbors(self, i):
        return self.adj_ls[i]
    
    def edge_list(self):
        return [(i,j) for i in self.adj for j in self.adj[i] if i<j]
    
    def edge_count(self):
        # Must divide by 2 to avoid repeated counts
        return np.count_nonzero(self.adj_m) / 2
    
    def find_comp(self, i):
        """Find the component that contains node i for a given network object"""
        c = set()
        q = [i]
        while len(q) > 0:
            j = q.pop()
            c.add(j)
            q += self.neighbors(j) - c  # python type overloading
        return c
    
    def deg(self,i):
        return np.array(len(self.neighbors(i)))
    
    def deg_dist(self):
        return np.array([self.deg(i) for i in range(self.n)])
    






















# # Define network object with adjacency lists
# class Network(object):
#     def __init__(self, n):
#         self.adj = {i:set() for i in range(n)} # dictionary whcih map each node to a set of connected adjacent nodes

#     def add_edge(self, i, j):
#         self.adj[i].add(j)
#         self.adj[i].add(i)
        
#     def neighbors(self, i):
#         return self.adj[i]
    
#     def edge_list(self):
#         return [(i,j) for i in self.adj for j in self.adj[i] if i<j]



# # Alternative definition with adjency matrix (np array)
# class Network_mat(object):
#     def __init__(self, n):
#         self.n = n
#         self.adj = np.zeros((n,n), dtype = bool)

#     def add_edge(self, i, j):
#         self.adj[i][j] = 1
#         self.adj[j][i] = 1

#     def neighbors(self, i):
#         # return [j for j in range(self.n) if self.adj[i][j]==1]
#         return np.nonzero(self.adj[i])[0]
    
#     def edge_list(self):
#         return [(i,j) for i in range(self.n) for j in self.neighbors(i) if i<j]

