import math
import unittest

import global_game_data
import graph_data
import pathing
from permutation import sjt_permutations, is_hamiltonian_cycle


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
        self.assertEqual(path[0], 0)  # start
        self.assertEqual(path[-1], 3)  # end
        self.assertIn(2, path)  # target
        self.assertTrue(self.is_valid_path(path))

    def test_bfs_path(self):
        path = pathing.get_bfs_path()
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0)  # start
        self.assertEqual(path[-1], 3)  # end
        self.assertIn(2, path)  # target
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
        self.assertTrue(is_hamiltonian_cycle(graph, path))

    def test_no_hamiltonian_cycle(self):
        graph = [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]
        path = [1]  
        self.assertFalse(is_hamiltonian_cycle(graph, path))



if __name__ == '__main__':
    unittest.main()
