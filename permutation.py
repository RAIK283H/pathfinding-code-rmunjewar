import graph_data

def sjt_permutations(n):
    perm = list(range(1, n)) 
    direction = [-1] * (n - 1)

    yield perm[:]

    while True:
        mobile = -1
        mobile_index = -1
        for i in range(len(perm)):
            if (direction[i] == -1 and i > 0 and perm[i] > perm[i - 1]) or \
               (direction[i] == 1 and i < len(perm) - 1 and perm[i] > perm[i + 1]):
                if mobile == -1 or perm[i] > perm[mobile]:
                    mobile = perm[i]
                    mobile_index = i

        if mobile == -1:
            break

        swap_index = mobile_index + direction[mobile_index]
        perm[mobile_index], perm[swap_index] = perm[swap_index], perm[mobile_index]
        direction[mobile_index], direction[swap_index] = direction[swap_index], direction[mobile_index]

        for i in range(len(perm)):
            if perm[i] > mobile:
                direction[i] *= -1

        yield perm[:] 

def is_hamiltonian_cycle(graph, cycle):
    n = len(graph)
    visited = set(cycle)
    if len(visited) != n:
        return False

    for i in range(len(cycle) - 1):
        if graph[cycle[i]][cycle[i + 1]] == 0:
            return False

    if graph[cycle[-1]][cycle[0]] == 0:
        return False

    return True

def find_hamiltonian_cycles(graph):
    n = len(graph)
    valid_cycles = []

    for perm in sjt_permutations(n - 1):
        cycle = [0] + perm + [n - 1]
        if is_hamiltonian_cycle(graph, cycle):
            valid_cycles.append(cycle)

    return valid_cycles if valid_cycles else None

def calculate_distance(graph, cycle):
    total_distance = 0
    for i in range(len(cycle) - 1):
        total_distance += graph[cycle[i]][cycle[i + 1]]
    return total_distance

def find_optimal_cycles(graph):
    n = len(graph)
    valid_cycles = []

    for perm in sjt_permutations(n - 2):
        cycle = [0] + perm + [n - 1]
        if is_hamiltonian_cycle(graph, cycle):
            distance = calculate_distance(graph, cycle)
            valid_cycles.append((cycle, distance))

    if not valid_cycles:
        return None

    min_distance = min(distance for _, distance in valid_cycles)
    optimal_cycles = [cycle for cycle, distance in valid_cycles if distance == min_distance]

    return optimal_cycles

def is_clique(graph, subset):
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if graph[subset[i]][subset[j]] == 0:
                return False
    return True

def find_largest_clique(graph):
    n = len(graph)
    nodes = list(range(1, n - 1))
    largest_clique = []

    def generate_subsets(subset, index):
        nonlocal largest_clique
        if is_clique(graph, subset) and len(subset) > len(largest_clique):
            largest_clique = subset[:]

        for i in range(index, len(nodes)):
            subset.append(nodes[i])
            generate_subsets(subset, i + 1)
            subset.pop()

    generate_subsets([], 0)

    return largest_clique