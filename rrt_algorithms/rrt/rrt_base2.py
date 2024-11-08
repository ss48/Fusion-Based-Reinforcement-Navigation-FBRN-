import random

import numpy as np
import sys
sys.path.append('/home/dell/rrt-algorithms')


from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.rrt.tree import Tree
from rrt_algorithms.utilities.geometry import steer


class RRTBase(object):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param q: length of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.q = q
        self.r = r
        self.prc = prc
        self.x_init = tuple(x_init)
        self.x_goal = tuple(x_goal)
        self.trees = []  # list of all trees
        self.add_tree()  # add initial tree

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(Tree(self.X))


    def nearby(self, tree, x, n):
        """
        Return nearby vertices
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :param n: int, max number of neighbors to return
        :return: list of nearby vertices
        """
        return self.trees[tree].V.nearest(x, num_results=n, objects="raw")



    def connect_to_point(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False


    def can_connect_to_goal(self, tree):
        """
        Check if the goal can be connected to the graph.
        :param tree: rtree of all vertices
        :return: True if can be added, False otherwise
        """
        x_nearest = self.get_nearest(tree, self.x_goal)

    # Ensure x_goal is in the edge dictionary and has neighbors
    # Use an empty dictionary as the fallback only if the default should be iterable
        edges_to_goal = self.trees[tree].E.get(self.x_goal, {})
    
        if edges_to_goal is not None and isinstance(edges_to_goal, dict) and x_nearest in edges_to_goal:
        # Goal is already connected using the nearest vertex
            return True

    # Check if obstacle-free and connectable
        if self.X.collision_free(x_nearest, self.x_goal, self.r):
            return True
    
        return False



    def get_path(self):
        """
        Return path through tree from start to goal
        :return: path if possible, None otherwise
        """
        if self.can_connect_to_goal(0):
            print("Can connect to goal")
            self.connect_to_goal(0)
            return self.reconstruct_path(0, self.x_init, self.x_goal)
        print("Could not connect to goal")
        return None



    def check_solution(self):
        # probabilistically check if solution found
        if self.prc and random.random() < self.prc:
            print("Checking if can connect to goal at", str(self.samples_taken), "samples")
            path = self.get_path()
            if path is not None:
                return True, path
        # check if can connect to goal after generating max_samples
        if self.samples_taken >= self.max_samples:
            return True, self.get_path()
        return False, None
    
    def add_vertex(self, tree, v):
        # Convert v to a tuple if it’s a numpy array
        v = tuple(v)
        self.trees[tree].V.insert(0, v + v, v)
        self.trees[tree].V_count += 1
        self.samples_taken += 1

    def add_edge(self, tree, child, parent):
        child = tuple(child)
        parent = tuple(parent) if parent is not None else None
        self.trees[tree].E[child] = parent

    def get_nearest(self, tree, x):
        return tuple(next(self.nearby(tree, tuple(x), 1)))

    def new_and_near(self, tree, q):
        x_rand = tuple(self.X.sample_free())
        x_nearest = tuple(self.get_nearest(tree, x_rand))
        x_new = self.bound_point(steer(x_nearest, x_rand, q))
        if not self.trees[0].V.count(x_new) == 0 or not self.X.obstacle_free(x_new):
            return None, None
        self.samples_taken += 1
        return tuple(x_new), tuple(x_nearest)

    def connect_to_goal(self, tree):
        x_nearest = tuple(self.get_nearest(tree, self.x_goal))
        self.trees[tree].E[self.x_goal] = x_nearest

    def reconstruct_path(self, tree, x_init, x_goal):
        x_init, x_goal = tuple(x_init), tuple(x_goal)
        path = [x_goal]
        current = x_goal
        while current != x_init:
            parent = tuple(self.trees[tree].E[current])
            path.append(parent)
            current = parent
        path.reverse()
        return path

    def bound_point(self, point):
        point = np.maximum(point, self.X.dimension_lengths[:, 0])
        point = np.minimum(point, self.X.dimension_lengths[:, 1])
        return tuple(point)  # Ensure bound points are tuples

