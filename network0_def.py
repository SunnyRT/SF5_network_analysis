import numpy as np

# Define network object with implementation of numpy arrays
class Network(object):
    def __init__(self, n=None, adj_m=None):
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
        self.adj_ls[i].add(j)
        self.adj_ls[j].add(i)
        self.adj_m[i][j] = 1
        self.adj_m[j][i] = 1

    def remove_edge(self, i, j):
        self.adj_ls[i].remove(j)
        self.adj_ls[j].remove(i)
        self.adj_m[i][j] = 0
        self.adj_m[j][i] = 0


    def neighbors(self, i):
        return self.adj_ls[i]

    
    def edge_list(self, shuffle=False):
        edge_ls = [(i,j) for i in range(self.n) for j in self.adj_ls[i] if i<j]
        edge_ls = np.array(edge_ls) # convert to numpy array for shuffling
        if shuffle:
            edge_ls_shuffled = np.random.permutation(edge_ls)
            return edge_ls_shuffled
        else:
            return edge_ls
    
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
    
    def comp_size(self,i):
        return len(self.find_comp(i))
    
    def deg(self,i):
        return np.array(len(self.neighbors(i)))
    
    def deg_dist(self):
        return np.array([self.deg(i) for i in range(self.n)])
    
    def remove_nodes(self, i_ary):
        """Remove nodes with index from i_ary and create a new network object.
        Create a new copy of network object with nodes removed."""
        mask = np.ones(self.n, dtype=bool)
        mask[i_ary] = False
        new_adj_m = self.adj_m[mask][:, mask]

        return Network(adj_m=new_adj_m)   








class Network_dir(object): # Directed network object
    def __init__(self, n):
        self.n = n
        self.adj_ls = np.empty(n, dtype=object)
        self.adj_ls.fill(set())
        self.adj_m = np.zeros((n,n), dtype = bool)

    def add_edge(self, i, j): # From i to j
        self.adj_ls[i].add(j)
        self.adj_m[i][j] = 1


    def remove_edge(self, i, j): # From i to j
        self.adj_ls[i].remove(j)
        self.adj_m[i][j] = 0


    def neighbors_sink(self, i): # Outgoing edges i.e. all neighbors j that can be reached from node i
        return self.adj_ls[i]
    
    def neighbors_source(self, i): # Incoming edges i.e. all neighbors j that can reach node i
        return np.array(set(np.nonzero(self.adj_m[:,i])[0]))

    
    def edge_list(self, shuffle=False):
        edge_ls = [(i,j) for i in range(self.n) for j in self.adj_ls[i]] # Directed edges from i to j. i.e. (source, sink)
        edge_ls = np.array(edge_ls) # convert to numpy array for shuffling
        if shuffle:
            edge_ls = np.random.permutation(edge_ls)
        return edge_ls
    
    def edge_count(self):
        # Must NOT divide by 2 since (i,j) and (j,i) are distinct
        return np.count_nonzero(self.adj_m) 
    
    def find_sink(self, i):
        """Find the sink component, which is the collection of all nodes reachable by node i for a given network object"""
        c = set()
        q = [i]
        while len(q) > 0:
            j = q.pop()
            c.add(j)
            q += self.neighbors_sink(j) - c

        return c
    
    def sink_size(self,i):
        return len(self.find_sink(i))



