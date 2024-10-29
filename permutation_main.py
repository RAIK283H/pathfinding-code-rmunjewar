import graph_data
from permutation import find_hamiltonian_cycles
from permutation import find_optimal_cycles
from permutation import find_largest_clique

# exmaple graph to test
graph = [
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0]
]

find_optimal_cycles(graph)

find_largest_clique(graph)

find_hamiltonian_cycles(graph)