import numpy as np
import graph_data
import global_game_data

def convert_adjancy_list_to_matrix(graph):
    n = len(graph)
    matrix = np.full((n,n), np.inf)
    np.fill_diagonal(matrix, 0)
    
    # all nodes
    for node, (_, neighbors) in enumerate(graph):
        for neighbor in neighbors:
            matrix[node, neighbor] = 1 
    
    return matrix

def floyd_warshall(graph_matrix):
    n = graph_matrix.shape[0]
    dist = graph_matrix.copy()
    parent = np.full((n, n), -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if dist[i, j] != np.inf and i != j:
                parent[i, j] = i
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    parent[i, j] = parent[k, j]
    
    return dist, parent

def build_path(parent_matrix, start, end):
    if parent_matrix[start, end] == -1:
        return None  
    
    path = [end]
    current = end
    
    while current != start:
        current = parent_matrix[start, current]
        path.insert(0, current)
    
    return path

def get_floyd_warshall_path():
    graph = graph_data.graph_data[global_game_data.current_graph_index]
    start_node = 0
    target_node = global_game_data.target_node[global_game_data.current_graph_index]
    end_node = len(graph) - 1

    graph_matrix = convert_adjacency_list_to_matrix(graph)
    
    dist_matrix, parent_matrix = floyd_warshall(graph_matrix)
    
    path_to_target = build_path(parent_matrix, start_node, target_node)
    path_from_target = build_path(parent_matrix, target_node, end_node)
    
    if path_to_target and path_from_target:
        full_path = path_to_target + path_from_target[1:]
        
        assert full_path[0] == start_node, "Result path must begin at the start node"
        assert full_path[-1] == end_node, "Result path must end at the exit node"
        assert target_node in full_path, "Result path must include the target node"
        assert all(full_path[i+1] in graph[full_path[i]][1] for i in range(len(full_path)-1)), \
            "Every pair of sequential vertices in the path must be connected by an edge"
        
        return full_path
    else:
        return get_floyd_warshall_path()
