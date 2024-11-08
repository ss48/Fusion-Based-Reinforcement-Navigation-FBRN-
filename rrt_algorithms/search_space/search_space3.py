# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#the RRT and DWA algorithms function with the optimized collision checking using Bounding Volume Hierarchies (BVH).
#BVH Implementation: We use an R-tree (a type of BVH) to store axis-aligned bounding boxes (AABB) for each obstacle. This allows us to quickly identify which obstacles might intersect with a given point or path segment.
#Efficient Collision Checking: Instead of checking every obstacle for collision, we first use the BVH to narrow down the list to only those obstacles whose bounding boxes intersect with the point or path segment in question.
#Integration with DWA: The code remains the same in terms of how paths are generated and optimized, but now the collision checking is more efficient, leading to faster overall performance.
#The drone is modeled as a sphere with a 10 cm radius (0.1 meters). This radius is added to the radius of each obstacle during collision detection.
#Expanded Obstacles: By expanding each obstacle's radius during collision checks, we ensure that the drone is considered in the obstacle avoidance logic, avoiding potential collisions.
#all_obstacles as a combination of the obstacles list (static obstacles) and Obstacles (randomly generated obstacles).
import numpy as np
from rtree import index

from rrt_algorithms.utilities.geometry import es_points_along_line
from rrt_algorithms.utilities.obstacle_generation2 import obstacle_generator

from rtree import index

class SearchSpace:
    def __init__(self, dimension_lengths, obstacles=None, buffer=0.1):
        self.dimension_lengths = dimension_lengths
        self.obstacles = obstacles if obstacles is not None else []
        self.dimensions = len(dimension_lengths)
        self.drone_radius = 0.10  # Drone radius in meters (10 cm for a 20 cm diameter drone)
        self.buffer = buffer  # Set buffer around obstacles
        self.build_bvh()

    
    
    
    def build_bvh(self):
        p = index.Property()
        p.dimension = 3  # Set dimension to 3 for 3D bounding boxes
        self.bvh = index.Index(properties=p)
        for i, obstacle in enumerate(self.obstacles):
            aabb = self.compute_aabb(obstacle)
            self.bvh.insert(i, aabb)

    def compute_aabb(self, obstacle):
        x_center, y_center, z_min, z_max, radius = obstacle
        radius += self.drone_radius + self.buffer  # Expand obstacle radius
        x_min = x_center - radius
        x_max = x_center + radius
        y_min = y_center - radius
        y_max = y_center + radius
        z_min = z_min  # Lower z-bound
        z_max = z_max  # Upper z-bound
        return (x_min, y_min, z_min, x_max, y_max, z_max)
        
    def is_point_in_obstacle(self, point, obstacle):
        x_center, y_center, z_min, z_max, radius = obstacle
        effective_radius = radius + self.drone_radius + self.buffer  # Add buffer
        dist_xy = np.linalg.norm(np.array(point[:2]) - np.array([x_center, y_center]))
        within_z_bounds = z_min <= point[2] <= z_max
        return dist_xy <= effective_radius and within_z_bounds

    def intersects_obstacle(self, point, obs):
        x, y, z = point
        x_center, y_center, z_min, z_max, radius = obs
        in_cylinder = (x - x_center)**2 + (y - y_center)**2 <= radius**2
        within_height = z_min <= z <= z_max
        return in_cylinder and within_height
        
    def obstacle_free(self, point):
        # Ensure that point has at least three elements for x, y, z
        if len(point) < 3:
            raise ValueError(f"Point {point} is incorrectly formatted, expected at least 3 elements for (x, y, z).")
        
        # Extract spatial coordinates
        point = point[:3]

        nearby_obstacles = list(self.bvh.intersection((point[0], point[1], point[2], point[0], point[1], point[2])))
        for obs_idx in nearby_obstacles:
            if self.is_point_in_obstacle(point, self.obstacles[obs_idx]):
                return False
        return True

    def collision_free(self, point1, point2=None, steps=10):
        if point2 is None:
        # Single point collision check
            return all(not self.intersects_obstacle(point1, obs) for obs in self.obstacles)
    
    # Convert steps to an integer to avoid TypeErrors
        steps = int(steps) if isinstance(steps, (int, float)) else 10  # Default to 10 if steps is not specified correctly

    # Check multiple points along the segment for collision
        for t in np.linspace(0, 1, steps):
            intermediate_point = (1 - t) * np.array(point1) + t * np.array(point2)
            if any(self.intersects_obstacle(intermediate_point, obs) for obs in self.obstacles):
                return False
        return True

    def build_obs_index(self):
        # Create an R-tree index for obstacles
        p = index.Property()
        p.dimension = 3  # Set dimension to 3 for 3D obstacles
        self.obs = index.Index(properties=p)
        for i, obstacle in enumerate(self.obstacles):
            aabb = self.compute_aabb(obstacle)
            self.obs.insert(i, aabb)
            
 

    def sample(self):
        """
        Return a random location within the search space dimensions.
        :return: A tuple representing a random location within the search space.
        """
        return tuple(np.random.uniform(low, high) for low, high in self.dimension_lengths)




    def es_points_along_line(self, x_a, x_b, r):
        """
        Generate points along a line segment between x_a and x_b.
        :param x_a: Starting point of the line segment.
        :param x_b: Ending point of the line segment.
        :param r: Resolution for sampling points.
        :return: List of points along the line segment.
        """
        distance = np.linalg.norm(np.array(x_b) - np.array(x_a))
        num_points = int(distance / r) + 1
        return [tuple(np.array(x_a) + t * (np.array(x_b) - np.array(x_a))) for t in np.linspace(0, 1, num_points)]




    def sample_free(self):
        while True:
            point = self.sample()
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 3:
            # Check if point is free of obstacles
                if self.obstacle_free(point):
                    return point
            else:
                print(f"Sampled point {point} is incorrectly formatted. Expected at least 3 elements.")


