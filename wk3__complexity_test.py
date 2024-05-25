import numpy as np
import time

def SIR_sets(network, lambda_, seed_n, Nr_only=False):
    n = network.n
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

def SIR_numpy(network, lambda_, seed_n, Nr_only=False):
    n = network.n
    S = np.ones(n)
    I = np.zeros(n)
    R = np.zeros(n)

    seeds = np.random.choice(n, seed_n, replace=False)
    S[seeds] = 0
    I[seeds] = 1

    Ns_t = [n - seed_n]
    Ni_t = [seed_n]
    Nr_t = [0]

    neighbors = [list(network.neighbors(i)) for i in range(n)]

    while np.sum(I) > 0:
        new_I = np.zeros(n)
        new_R = np.zeros(n)
        for i in range(n):
            if I[i] == 1:
                for j in neighbors[i]:
                    if S[j] == 1 and np.random.rand() < lambda_:
                        S[j] = 0
                        new_I[j] = 1
                I[i] = 0
                R[i] = 1
        I = new_I
        R += new_R
        Ns_t.append(np.sum(S))
        Ni_t.append(np.sum(I))
        Nr_t.append(np.sum(R))

    if Nr_only:
        return Nr_t[-1]
    return Ns_t, Ni_t, Nr_t

# Dummy network for testing
class DummyNetwork:
    def __init__(self, n):
        self.n = n
        self.adj_ls = [np.random.choice(n, size=(np.random.randint(1, 5)), replace=False) for _ in range(n)]
    def neighbors(self, i):
        return self.adj_ls[i]

# Testing
network = DummyNetwork(10000)
lambda_ = 0.1
seed_n = 1

# Sets approach
start_time = time.time()
for _ in range(10):
    SIR_sets(network, lambda_, seed_n)
print("Sets approach time:", time.time() - start_time)

# NumPy approach
start_time = time.time()
for _ in range(10):
    SIR_numpy(network, lambda_, seed_n)
print("NumPy approach time:", time.time() - start_time)