import numpy as np
import random
import time
import networkx as nx
from pyqubo import Array
from qubobrute.core import *
from qubobrute.simulated_annealing import simulate_annealing_gpu
from pyqubo import Spin

G = nx.complete_graph(4)
G = nx.convert_node_labels_to_integers(G)
N = nx.number_of_nodes(G)

J = nx.adjacency_matrix(G).todense()

for i in range(N):
    for j in range(i, N):
        if J[i, j] == 1:
            J[i, j] = random.choice([1, -1])
            J[j, i] = J[i, j]


def qubo_energy(qubo: np.ndarray, offset: np.number, sample: np.ndarray) -> np.number:
    """Calculate the energy of a sample."""
    return np.dot(sample, np.dot(qubo, sample)) + offset


# Define binary variables
spins = [Spin(f'spin_{i}') for i in range(N)]

# Construct the Hamiltonian
H = 0.5 * np.sum(J * np.outer(spins, spins))

# Compile the model to a binary quadratic model (BQM)
model = H.compile()
qubo, offset = model.to_qubo(index_label=True)

# Determine the shape of the array (assuming you have all the indices)
max_row = max(index[0] for index in qubo.keys()) + 1
max_col = max(index[1] for index in qubo.keys()) + 1

# Initialize the 2D NumPy array with zeros
q = np.zeros((max_row, max_col))

# Fill the array with the values from the dictionary
for index, value in qubo.items():
    q[index] = value

start = time.time()
energies, solutions = simulate_annealing_gpu(q, offset, n_iter=1000, n_samples=1000, temperature=1.0, cooling_rate=0.99)

# Find the minimum energy
min_energy = energies.min()

# Find all indices with the minimum energy
min_indices = np.where(energies == min_energy)[0]

# Create a set to store unique solutions
unique_solutions = set()

# Print all unique solutions with the minimum energy
print("SA Solutions: ")
print("Minimum Energy: ", min_energy)
for index in min_indices:
    # Convert the solution to a tuple to make it hashable
    solution_tuple = tuple(solutions[index])
    if solution_tuple not in unique_solutions:
        unique_solutions.add(solution_tuple)

print("SA time taken: ", time.time() - start)

print(f"Found {len(unique_solutions)} solutions:")
print(unique_solutions)

# assign an equal probability of finding each of the ground states
prob = 1/len(unique_solutions)

# Convert the list of lists to a 2D NumPy array
unique_solutions_np = np.array([list(tup) for tup in unique_solutions])

C = np.corrcoef(unique_solutions_np, rowvar=False)

scores = np.sum(C, axis=1) - 1

print(scores)


start = time.time()

# brute-force
energies = solve_gpu(q, offset)

# Find the minimum energy
min_energy = energies.min()

# Find all indices with the minimum energy
min_indices = np.where(energies == min_energy)[0]

# Create a set to store unique solutions
unique_solutions = set()

# Print all unique solutions with the minimum energy
print("Brute Force Solutions: ")
print("Minimum Energy: ", min_energy)
for index in min_indices:
    # Get the solution bits for the current index
    solution = bits(index, nbits=N)

    # Convert the solution to a tuple to make it hashable
    solution_tuple = tuple(solution)

    # Check if the solution is unique
    if solution_tuple not in unique_solutions:
        unique_solutions.add(solution_tuple)
        # Calculate the QUBO energy for the solution
        energy = qubo_energy(q, offset, sample=solution)

print("Brute Force time taken: ", time.time() - start)
print(f"Found {len(unique_solutions)} solutions:")
print(unique_solutions)