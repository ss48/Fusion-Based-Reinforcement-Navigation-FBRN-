import numpy as np
from rrt_algorithms.rrt.rrt_base2 import RRTBase


class RRT(RRTBase):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)

    def rrt_search(self):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            print(f"q: {self.q}")
            x_new, x_nearest = self.new_and_near(0, self.q)
            print(f"x_new: {x_new}, x_nearest: {x_nearest}")

            if x_new is None:
                continue

            # connect shortest valid edge
            self.connect_to_point(0, x_nearest, x_new)

            solution = self.check_solution()
            if solution[0]:
                return solution[1]
    
    def add_edge(self, tree, child, parent):
        # Convert numpy arrays to tuples if they are used as keys
        if isinstance(child, np.ndarray):
            child = tuple(child)
        if isinstance(parent, np.ndarray) and parent is not None:
            parent = tuple(parent)
            
        # Assign the edge in the tree's edge dictionary
        self.trees[tree].E[child] = parent