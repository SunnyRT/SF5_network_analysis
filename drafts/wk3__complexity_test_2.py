import numpy as np
import timeit

def generate_zeros_and_ones_choice(size, prob):
    return np.random.choice([0, 1], size=size, p=[1-prob, prob])

def generate_zeros_and_ones_binomial(size, prob):
    return np.random.binomial(1, prob, size)

# Define parameters
size = 1000000
lambda_ = 0.3  # Probability of getting a 1

# Time the `numpy.random.choice` method
choice_time = timeit.timeit(lambda: generate_zeros_and_ones_choice(size, lambda_), number=10)

# Time the `numpy.random.binomial` method
binomial_time = timeit.timeit(lambda: generate_zeros_and_ones_binomial(size, lambda_), number=10)

print(f"Time using numpy.random.choice: {choice_time:.6f} seconds")
print(f"Time using numpy.random.binomial: {binomial_time:.6f} seconds")