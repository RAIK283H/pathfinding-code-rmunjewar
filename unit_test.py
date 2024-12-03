import math
import unittest

import numpy as np

import global_game_data
import graph_data
import pathing
from permutation import calculate_distance, find_hamiltonian_cycles, find_largest_clique, find_optimal_cycles, is_clique, sjt_permutations, is_hamiltonian_cycle
from f_w import (
    convert_adjancy_list_to_matrix, 
    floyd_warshall, 
    build_path, 
    get_floyd_warshall_path
)

class TestPathFinding(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('test'.upper(), 'TEST')

    def test_isupper(self):
        self.assertTrue('TEST'.isupper())
        self.assertFalse('Test'.isupper())

    def test_floating_point_estimation(self):
        first_value = 0
        for x in range(1000):
            first_value += 1/100
        second_value = 10
        almost_pi = 3.1
        pi = math.pi
        self.assertNotEqual(first_value,second_value)
        self.assertAlmostEqual(first=first_value,second=second_value,delta=1e-9)
        self.assertNotEqual(almost_pi, pi)
        self.assertAlmostEqual(first=almost_pi, second=pi, delta=1e-1)

    def setUp(self):
        global_game_data.current_graph_index = 0
        graph_data.graph_data = [
            [((0, 0), [1, 2]), ((1, 1), [0, 3]), ((2, 2), [0, 3]), ((3, 3), [1, 2, 4]), ((4, 4), [3])]
        ]
        global_game_data.target_node = [2]

    def setUp(self):
        self.graph1 = [
            (0, [1]),    
            (1, [0, 2]), 
            (2, [1, 3]), 
            (3, [2])    
        ]
        
        self.graph2 = [
            (0, [1, 2]),
            (1, [0, 3]), 
            (2, [0, 3]), 
            (3, [1, 2]) 
        ]
        
        
        graph_data.graph_data = [self.graph1, self.graph2]
        global_game_data.current_graph_index = 0
        global_game_data.target_node = [2, 3]  

    def test_random_path_basic_graph(self): # testing on first graph
        global_game_data.current_graph_index = 0
        path = pathing.get_random_path()

        self.assertEqual(path[0], 0, "Path must start at node 0")
        self.assertEqual(path[-1], 3, "Path must end at the last node")
        self.assertIn(2, path, "Path must include the target node")
        graph = graph_data.graph_data[global_game_data.current_graph_index]
        for i in range(len(path) - 1):
            self.assertIn(path[i + 1], graph[path[i]][1], "Sequential nodes in the path must be connected")

    def test_dfs_path(self):
        path = pathing.get_dfs_path()
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0) 
        self.assertEqual(path[-1], 3)  
        self.assertIn(2, path) 
        self.assertTrue(self.is_valid_path(path))

    def test_bfs_path(self):
        path = pathing.get_bfs_path()
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0)  
        self.assertEqual(path[-1], 3) 
        self.assertIn(2, path)  
        self.assertTrue(self.is_valid_path(path))

    def test_dfs_path_different_start_target_end(self):
        global_game_data.current_graph_index = 1
        path = pathing.get_dfs_path()
        self.assertEqual(path[0], 0, "DFS path must start at node 0")
        self.assertEqual(path[-1], 3, "DFS path must end at the last node")

        self.assertIn(3, path, "DFS path must include the target node")

        graph = graph_data.graph_data[global_game_data.current_graph_index]
        for i in range(len(path) - 1):
            self.assertIn(path[i + 1], graph[path[i]][1], "Sequential nodes in the DFS path must be connected")

    def test_bfs_path_different_start_target_end(self):
        global_game_data.current_graph_index = 1
        path = pathing.get_bfs_path()
        self.assertEqual(path[0], 0, "BFS path must start at node 0")
        self.assertEqual(path[-1], 3, "BFS path must end at the last node")
        self.assertIn(3, path, "BFS path must include the target node")
        graph = graph_data.graph_data[global_game_data.current_graph_index]
        for i in range(len(path) - 1):
            self.assertIn(path[i + 1], graph[path[i]][1], "Sequential nodes in the BFS path must be connected")


    def is_valid_path(self, path):
        graph = graph_data.graph_data[global_game_data.current_graph_index]
        return all(path[i+1] in graph[path[i]][1] for i in range(len(path)-1))

class TestPermutation(unittest.TestCase):
    def setUp(self):
        self.graph1 = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        
        self.graph2 = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]

        self.complete_graph = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ]
        
        self.cycle_graph = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]
        
        self.disconnected_graph = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        
        self.single_node_graph = [[0]]
        
        self.two_node_graph = [
            [0, 1],
            [1, 0]
        ]
    def test_sjt_permutations(self):
        perms = list(sjt_permutations(3))
        expected = [[1, 2], [2, 1]]
        self.assertEqual(perms, expected)

    def test_is_hamiltonian_cycle(self):
        graph = [
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0]
        ]
        path = [1, 2] 
        self.assertFalse(is_hamiltonian_cycle(graph, path))

    def test_no_hamiltonian_cycle(self):
        graph = [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]
        path = [1]  
        self.assertFalse(is_hamiltonian_cycle(graph, path))

    def test_calculate_distance(self):
            cycle = [0, 1, 2, 0]
            expected_distance = 3  
            self.assertEqual(calculate_distance(self.graph1, cycle), expected_distance)

    def test_find_optimal_cycles(self):
            optimal_cycles = find_optimal_cycles(self.graph1)
            optimal_cycle = [0, 1, 2]
            optimal_distance = 3
            self.assertIn((optimal_cycle, optimal_distance), optimal_cycles)

    def test_is_clique(self):
            subset = [0, 1, 2]
            self.assertTrue(is_clique(self.graph1, subset))
            
            subset = [0, 2]
            self.assertFalse(is_clique(self.graph2, subset))

    def test_sjt_empty_permutation(self):
        """Test SJT with n=1 (should yield empty list)"""
        perms = list(sjt_permutations(1))
        self.assertEqual(perms, [[]])

    def test_hamiltonian_cycle_complete_graph(self):
        """Test finding Hamiltonian cycles in a complete graph"""
        cycles = find_hamiltonian_cycles(self.complete_graph)
        self.assertIsNotNone(cycles)
        for cycle in cycles:
            self.assertTrue(is_hamiltonian_cycle(self.complete_graph, cycle))

    def test_hamiltonian_cycle_cycle_graph(self):
        cycles = find_hamiltonian_cycles(self.cycle_graph)
        self.assertIsNotNone(cycles)
        self.assertTrue(any(is_hamiltonian_cycle(self.cycle_graph, cycle) for cycle in cycles))

    def test_single_node_hamiltonian(self):
        self.assertFalse(is_hamiltonian_cycle(self.single_node_graph, [0]))

    def test_two_node_hamiltonian(self):
        self.assertTrue(is_hamiltonian_cycle(self.two_node_graph, [0, 1]))

    def test_optimal_cycles_complete_graph(self):
        optimal_cycles = find_optimal_cycles(self.complete_graph)
        self.assertIsNotNone(optimal_cycles)
        distances = [dist for _, dist in optimal_cycles]
        self.assertEqual(len(set(distances)), 1)

    def test_is_clique_empty_set(self):
        self.assertTrue(is_clique(self.graph1, []))

    def test_is_clique_single_vertex(self):
        self.assertTrue(is_clique(self.graph1, [1]))

    def test_clique_disconnected_graph(self):
        clique = find_largest_clique(self.disconnected_graph)
        self.assertTrue(len(clique) <= 2)

    def test_invalid_cycle_repeated_nodes(self):
        cycle = [0, 1, 1, 2]
        self.assertFalse(is_hamiltonian_cycle(self.graph1, cycle))

    def test_invalid_cycle_missing_nodes(self):
        cycle = [0, 1]
        self.assertFalse(is_hamiltonian_cycle(self.graph1, cycle))

class TestAdvancedPathfinding(unittest.TestCase):
    def setUp(self):
        self.simple_graph = [
            ((0, 0), [1, 2]),     
            ((1, 1), [0, 3]),       
            ((2, 2), [0, 3]),       
            ((3, 3), [1, 2, 4]),    
            ((4, 4), [3])          
        ]
        
        self.complex_graph = [
            ((0, 0), [1, 2, 3]), 
            ((1, 1), [0, 2, 4]),
            ((2, 2), [0, 1, 3, 4]),
            ((3, 3), [0, 2, 4]),
            ((4, 4), [1, 2, 3])
        ]

        self.graph = [
            ((0, 0), [1, 2]), 
            ((1, 1), [0, 2, 3]),
            ((2, 2), [0, 1, 3]), 
            ((3, 3), [1, 2]),
        ]
        graph_data.graph_data = [self.graph]
        global_game_data.current_graph_index = 0
        global_game_data.target_node = [3]

        graph_data.graph_data = [self.simple_graph, self.complex_graph]
        graph_data.graph_coordinates = {
            0: (0, 0),
            1: (1, 1),
            2: (2, 2),
            3: (3, 3),
            4: (4, 4)
        }
        
    def test_dijkstra_simple_graph(self):
        global_game_data.current_graph_index = 0
        global_game_data.target_node = [2] 
        
        path = pathing.get_dijkstra_path()
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0, "Path must start at node 0")
        self.assertEqual(path[-1], 4, "Path must end at node 4")
        self.assertIn(2, path, "Path must contain target node")
        
        for i in range(len(path) - 1):
            self.assertIn(path[i + 1], self.simple_graph[path[i]][1],
                         f"Nodes {path[i]} and {path[i + 1]} must be connected")

        
    def test_dijkstra_no_path(self):
        disconnected_graph = [
            ((0, 0), [1]),
            ((1, 1), [0]),
            ((2, 2), [3]),
            ((3, 3), [2]),
            ((4, 4), [])
        ]
        graph_data.graph_data = [disconnected_graph]
        global_game_data.current_graph_index = 0
        global_game_data.target_node = [2]
        
        with self.assertRaises(Exception):
            pathing.get_dijkstra_path()
            
    def test_astar_no_path(self):
        disconnected_graph = [
            ((0, 0), [1]),
            ((1, 1), [0]),
            ((2, 2), [3]),
            ((3, 3), [2]),
            ((4, 4), [])
        ]
        graph_data.graph_data = [disconnected_graph]
        global_game_data.current_graph_index = 0
        global_game_data.target_node = [2]
        
        with self.assertRaises(Exception):
            pathing.get_a_star_path()
    
    def test_dijkstra_correctness(self):
        path = pathing.get_dijkstra_path()
        
        self.assertEqual(path[0], 0, "Dijkstra path should start at node 0")
        self.assertEqual(path[-1], 4, "Dijkstra path should end at node 4")
        
        self.assertIn(1, path, "Dijkstra path should pass through node 1")
        self.assertIn(3, path, "Dijkstra path should pass through node 3")
        
        self.assertGreater(len(path), 1, "Dijkstra path should contain more than one node")

    def test_a_star_correctness(self):
        path = pathing.get_a_star_path()
        
        self.assertEqual(path[0], 0, "A* path should start at node 0")
        self.assertEqual(path[-1], 4, "A* path should end at node 4")
        
        self.assertIn(3, path, "A* path should pass through node 3")
        self.assertIn(2, path, "A* path should pass through node 2")
        
        self.assertGreater(len(path), 1, "A* path should contain more than one node")
    
    def test_dijkstra_vs_a_star(self):
        dijkstra_path = pathing.get_dijkstra_path()
        a_star_path = pathing.get_a_star_path()

        self.assertNotEqual(dijkstra_path, a_star_path, "Dijkstra and A* paths should not be the same for the same graph")

    def test_dijkstra_performance(self):
        import time
        start_time = time.time()
        path = pathing.get_dijkstra_path()
        end_time = time.time()
        self.assertTrue(end_time - start_time < 1, "Dijkstra took too long to run")
        
    def test_a_star_performance(self):
        import time
        start_time = time.time()
        path = pathing.get_a_star_path()
        end_time = time.time()
        self.assertTrue(end_time - start_time < 1, "A* took too long to run")

class TestFloydWarshall(unittest.TestCase):
    def setUp(self):
        self.simple_graph = [
            ((0, 0), [1, 2]),     
            ((1, 1), [0, 3]),       
            ((2, 2), [0, 3]),       
            ((3, 3), [1, 2, 4]),    
            ((4, 4), [3])          
        ]
        
        self.complex_graph = [
            ((0, 0), [1, 2, 3]), 
            ((1, 1), [0, 2, 4]),
            ((2, 2), [0, 1, 3, 4]),
            ((3, 3), [0, 2, 4]),
            ((4, 4), [1, 2, 3])
        ]
        
        self.disconnected_graph = [
            ((0, 0), [1]),
            ((1, 1), [0]),
            ((2, 2), [3]),
            ((3, 3), [2]),
            ((4, 4), [])
        ]

    def test_convert_adjacency_list_to_matrix(self):
        graph = self.simple_graph
        matrix = convert_adjancy_list_to_matrix(graph)
        
        self.assertEqual(matrix.shape, (len(graph), len(graph)))
        
        self.assertTrue(np.all(np.diag(matrix) == 0))
   
        x0, y0 = graph[0][0]
        x1, y1 = graph[1][0]
        expected_distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        self.assertAlmostEqual(matrix[0, 1], expected_distance)
        self.assertAlmostEqual(matrix[1, 0], expected_distance)
        self.assertTrue(np.isinf(matrix[0, 4]))

    def test_fw_baisc(self):
        graph_matrix = convert_adjancy_list_to_matrix(self.simple_graph)
        dist_matrix, parent_matrix = floyd_warshall(graph_matrix)
        
        self.assertEqual(dist_matrix.shape, (len(self.simple_graph), len(self.simple_graph)))
        self.assertEqual(parent_matrix.shape, (len(self.simple_graph), len(self.simple_graph)))
        
        self.assertTrue(np.all(np.diag(dist_matrix) == 0)) 
        self.assertTrue(np.all(dist_matrix >= 0)) 

    def test_get_fw_path(self):
        graph_data.graph_data = [self.simple_graph, self.complex_graph]
        global_game_data.current_graph_index = 0
        global_game_data.target_node = [2, 3]
        
        path = get_floyd_warshall_path()
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0) 
        self.assertEqual(path[-1], 4)  
        self.assertIn(2, path) 
        graph = self.simple_graph
        for i in range(len(path) - 1):
            self.assertIn(path[i + 1], graph[path[i]][1], 
                          f"Nodes {path[i]} and {path[i + 1]} must be connected")

    def test_fw_multiple_graphs(self):
        test_graphs = [self.simple_graph, self.complex_graph]
        
        for i, graph in enumerate(test_graphs):
            graph_data.graph_data = [graph]
            global_game_data.current_graph_index = 0
            global_game_data.target_node = [2]
            
            path = get_floyd_warshall_path()
            
            self.assertIsNotNone(path)
            self.assertEqual(path[0], 0)
            self.assertEqual(path[-1], len(graph) - 1)
            
            self.assertIn(2, path)

    def test_floyd_warshall_performance(self):
        import time
        graph_matrix = convert_adjancy_list_to_matrix(self.simple_graph)
        
        start_time = time.time()
        floyd_warshall(graph_matrix)
        end_time = time.time()
        
        self.assertTrue(end_time - start_time < 1, "Floyd-Warshall took too long to run")

if __name__ == '__main__':
    unittest.main()
