# Pathfinding Starter Code

## overview

This project implements a simple pathfinding system where two players are tasked with getting from the start node to the exit node in different graphs. The goal is to explore the different algorithms and graph structures through the different graphs.

## customer requirements

1. Random Player Path Generation

   - User Story: "The Random player should generate a randomly generated path which goes from the start to the exit, but also hits the target at some point between."
   - Derived Requirement: The random player will have a randomly assigned path which begins with the start node, ends with the exit node, and hits the target node along the path. TEach sequential pair of nodes in the path must be connected by an edge.

2. New Statistic on Scoreboard
   - User Story: "I want to see another statistic for each player on the scoreboard."
   - Derived Requirement: The scoreboard should display a new statistic for each player in addition to the distance traveled.

## random pathing algorithm

The random path generation algorithm for the Random Player works as follows:

1. Initial Setup:

   - The algorithm begins at the start node of the graph.
   - The target node is selected as a required checkpoint, and the exit node is the final destination.

2. Path Generation:

   - The algorithm generates a random path by selecting one of the adjacent nodes at each step.
   - The player is required to visit the target node before proceeding to the exit node
   - The algorithm ensures that each chosen node is connected by an edge to the next node in the path, following the graph's adjacency list.

3. Constraints:

   - The player cannot visit the exit node before reaching the target.
   - The algorithm avoids cycles by tracking visited nodes, ensuring the path does not backtrack unnecessarily or get stuck in infinite loops.
   - The player can wander between nodes, but if the path takes too long without reaching the target or exit, the algorithm reroutes to ensure progress.

4. Final Path:

   - The path ends when the player reaches the exit node, with a valid traversal from start to target, then from target to exit

## Nodes Visited

This statistic tracks the total number of unique nodes each player visits during their traversal from the start node to the exit node. This metric gives insight into how efficiently a player navigates through the graph:
Test Player: Typically follows a straight line and visits fewer nodes since it uses an ideal path.
Random Player: Tends to visit more nodes due to its random movements, often wandering before hitting the target and exit node.

Did work on extra credit by implementing A\* algorithm.

Extra credit - Also did work on switching out get_dijkstra_path() with get_floyd_warshall_path().

Note: The PNG is the same for the last algorithm, but a different color.
