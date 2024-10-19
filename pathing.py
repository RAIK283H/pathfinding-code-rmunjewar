import graph_data
import global_game_data
from numpy import random

def set_current_graph_paths():
    global_game_data.graph_paths.clear()
    global_game_data.graph_paths.append(get_test_path())
    global_game_data.graph_paths.append(get_random_path())
    global_game_data.graph_paths.append(get_dfs_path())
    global_game_data.graph_paths.append(get_bfs_path())
    global_game_data.graph_paths.append(get_dijkstra_path())


def get_test_path():
    return graph_data.test_path[global_game_data.current_graph_index]


def get_random_path():
    graph = graph_data.graph_data[global_game_data.current_graph_index]
    start_node = 0
    target_node = global_game_data.target_node[global_game_data.current_graph_index]
    end_node = len(graph) - 1

    def random_walk(start, end):
        path = [start]
        current = start
        visited = set([start])

        while current != end:
            neighbors = graph[current][1]
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if unvisited_neighbors:
                next_node = random.choice(unvisited_neighbors)
            elif neighbors:
                next_node = random.choice(neighbors)
            else:
                return None

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    path_to_target = random_walk(start_node, target_node)
    path_from_target = random_walk(target_node, end_node)

    assert path_to_target[0] == start_node, "Path must start at the start node"
    assert path_from_target[-1] == end_node, "Path must end at the exit node"

    if path_to_target and path_from_target:
        full_path = path_to_target + path_from_target[1:]
        assert target_node in full_path, "Path must include the target node"
        return full_path
    else:
        return get_random_path()

def get_dfs_path():
    graph = graph_data.graph_data[global_game_data.current_graph_index]
    start_node = 0
    target_node = global_game_data.target_node[global_game_data.current_graph_index]
    end_node = len(graph) - 1

    def dfs(start, end):
        stack = [(start, [start])]
        visited = set()

        while stack:
            (node, path) = stack.pop()
            if node not in visited:
                if node == end:
                    return path
                visited.add(node)
                for neighbor in reversed(graph[node][1]):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        return None

    path_to_target = dfs(start_node, target_node)
    path_from_target = dfs(target_node, end_node)

    if path_to_target and path_from_target:
        full_path = path_to_target + path_from_target[1:]
        
        # postconditions
        assert target_node in full_path, "Result path must include the target node"
        assert full_path[-1] == end_node, "Result path must end at the exit node"
        assert all(full_path[i+1] in graph[full_path[i]][1] for i in range(len(full_path)-1)), "Every pair of sequential vertices in the path must be connected by an edge"
        
        return full_path
    else:
        return get_dfs_path() # again


def get_bfs_path():
    graph = graph_data.graph_data[global_game_data.current_graph_index]
    start_node = 0
    target_node = global_game_data.target_node[global_game_data.current_graph_index]
    end_node = len(graph) - 1

    def bfs(start, end):
        queue = [(start, [start])]
        visited = set()

        while queue:
            (node, path) = queue.pop(0) # used a list
            if node not in visited:
                if node == end:
                    return path
                visited.add(node)
                for neighbor in graph[node][1]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        return None

    path_to_target = bfs(start_node, target_node)
    path_from_target = bfs(target_node, end_node)

    if path_to_target and path_from_target:
        full_path = path_to_target + path_from_target[1:]
        
        # postconditions
        assert target_node in full_path, "Result path must include the target node"
        assert full_path[-1] == end_node, "Result path must end at the exit node"
        assert all(full_path[i+1] in graph[full_path[i]][1] for i in range(len(full_path)-1)), "Every pair of sequential vertices in the path must be connected by an edge"
        
        return full_path
    else:
        return get_bfs_path() # again


def get_dijkstra_path():
    return [1,2]
