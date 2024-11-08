import numpy as np
import time
from rtree import index
import sys
import uuid
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append('/home/dell/rrt-algorithms')

from rrt_algorithms.rrt.rrt2 import RRT
from rrt_algorithms.search_space.search_space2 import SearchSpace
from rrt_algorithms.utilities.plotting2 import Plot
from rrt_algorithms.dwa_algorithm.DWA3 import DWA, FuzzyController
from rrt_algorithms.dqn_algorithm.DQN import DQNAgent
from rrt_algorithms.dqn_algorithm.DQN import DroneEnv
from rrt_algorithms.utilities.obstacle_generation2 import generate_random_cylindrical_obstacles 

# Define the search space dimensions
X_dimensions = np.array([(-100, 100), (-100, 100), (-100, 100)])
X = SearchSpace(X_dimensions)

def encode_2d(point):
    return (point[0], point[1])

# Original 3D points
x_init_3d = (-50, 50, -2)
x_goal_1_3d = (20, 20, 50)
x_goal_2_3d = (30, 40, -70)
x_goal_3_3d = (50, 60, 90)
x_goal_4_3d = (-70, 80, 30)
x_goal_5_3d = (-90, 20, -40)
x_goal_6_3d = (-60, 40, -80)
x_final_goal_3d = (0, -10, 0)

# 2D Encodings for rtree
x_init = encode_2d(x_init_3d)
x_goal_1 = encode_2d(x_goal_1_3d)
x_goal_2 = encode_2d(x_goal_2_3d)
x_goal_3 = encode_2d(x_goal_3_3d)
x_goal_4 = encode_2d(x_goal_4_3d)
x_goal_5 = encode_2d(x_goal_5_3d)
x_goal_6 = encode_2d(x_goal_6_3d)
x_final_goal = encode_2d(x_final_goal_3d)

# Generate random obstacles
n = 20  # Number of random obstacles
Obstacles = generate_random_cylindrical_obstacles(X, (0,0,0), (100,100,100), n)

# Define static obstacles
obstacles = [
    (10, 82, 0, 15, 15),  
    (80, 20, 0, 10, 19),   
#    (10, 30, 0, 50, 15),   
#    (80, 80, 0, 70, 22), 
#    (50, 50, 0, 50, 22),
]

# Combine static and randomly generated obstacles
all_obstacles = obstacles + Obstacles
X = SearchSpace(X_dimensions, all_obstacles)

# RRT parameters
q = 300
r = 1
max_samples = 3024
prc = 0.1

# DWA parameters
dwa_params = {
    'max_speed': 3.0,
    'min_speed': 0.0,
    'max_yaw_rate': 40.0 * np.pi / 180.0,
    'max_accel': 0.3,  
    'v_reso': 0.05,  
    'yaw_rate_reso': 0.2 * np.pi / 180.0,
    'dt': 0.1,
    'predict_time': 2.0,
    'to_goal_cost_gain': 1.5,
    'speed_cost_gain': 0.5,
    'obstacle_cost_gain': 1.0,
}
# Add this function at the beginning of the code or where helper functions are defined
# Modified function to match dimensions for 2D goal and 3D path points
# Adjusted find_closest_point to handle mixed 2D/3D points
def find_closest_point(path, goal):
    # Convert goal to 3D by appending z=0
    goal_3d = np.array([goal[0], goal[1], 0])
    
    distances = []
    for point in path:
        # Ensure each point is 3D by adding z=0 if it's 2D
        point_3d = np.array(point) if len(point) == 3 else np.array([point[0], point[1], 0])
        distances.append(np.linalg.norm(point_3d - goal_3d))
    
    return np.argmin(distances)


# Rest of the code remains the same

# Pathfinding functions
def compute_energy_usage(path, velocity):
    energy = 0.0
    for i in range(1, len(path)):
        distance = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        energy += distance * velocity
    return energy

def min_obstacle_clearance(path, obstacles):
    min_clearance = float('inf')
    for point in path:
        for obs in obstacles:
            clearance = np.linalg.norm(np.array(point[:2]) - np.array(obs[:2])) - obs[4]
            if clearance < min_clearance:
                min_clearance = clearance
    return min_clearance  

def path_length(path):
    return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))

def path_smoothness(path):
    total_curvature = 0.0
    for i in range(1, len(path) - 1):
        vec1 = np.array(path[i]) - np.array(path[i-1])
        vec2 = np.array(path[i+1]) - np.array(path[i])
        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        total_curvature += angle
    return total_curvature
# Adaptive Large Neighborhood Search (ALNS) Functions
def alns_optimize_path(path, X, all_obstacles, max_iterations=500):
    neighborhoods = [segment_change, detour_addition, direct_connection]
    neighborhood_scores = {segment_change: 1.0, detour_addition: 1.0, direct_connection: 1.0}
    current_path = path.copy()

    for _ in range(max_iterations):
        # Select neighborhood adaptively
        neighborhood = random.choices(neighborhoods, weights=neighborhood_scores.values())[0]
        new_path = neighborhood(current_path, X, all_obstacles)

        # Evaluate the new path
        new_path_length = path_length(new_path)
        current_path_length = path_length(current_path)

        if new_path_length < current_path_length:
            current_path = new_path
            neighborhood_scores[neighborhood] *= 1.1  # Reward successful neighborhood
        else:
            neighborhood_scores[neighborhood] *= 0.9  # Penalize unsuccessful neighborhood

    return current_path

def segment_change(path, X, all_obstacles, r=1):
    if len(path) < 3:
        return path

    i = random.randint(0, len(path) - 3)
    j = random.randint(i + 2, len(path) - 1)

    x_a = path[i]
    x_b = path[j]

    spatial_x_a = np.array(x_a[:3])
    spatial_x_b = np.array(x_b[:3])

    new_point_spatial = spatial_x_a + (spatial_x_b - spatial_x_a) / 2
    new_point = list(new_point_spatial) + list(x_a[3:])

    if X.collision_free(spatial_x_a, new_point_spatial, r) and X.collision_free(new_point_spatial, spatial_x_b, r):
        new_path = path[:i + 1] + [new_point] + path[j:]
        return new_path

    return path

def detour_addition(path, X, all_obstacles, r=1):
    if len(path) < 2:
        return path

    i = random.randint(0, len(path) - 2)
    x_a = path[i]
    x_b = path[i + 1]

    spatial_x_a = np.array(x_a[:3])
    spatial_x_b = np.array(x_b[:3])

    detour_point_3d = spatial_x_a + (np.random.rand(3) - 0.5) * 2 * r
    detour_point = list(detour_point_3d) + list(x_a[3:])

    if X.collision_free(spatial_x_a, detour_point_3d, r) and X.collision_free(detour_point_3d, spatial_x_b, r):
        new_path = path[:i + 1] + [detour_point] + path[i + 1:]
        return new_path

    return path


def direct_connection(path, X, all_obstacles, r=1):
    if len(path) < 3:
        return path  # Can't directly connect if there aren't enough points

    new_path = path.copy()
    i = random.randint(0, len(path) - 3)  # Select a random starting point
    j = random.randint(i + 2, len(path) - 1)  # Select a random ending point

    x_a = path[i][:3]  # Extract only the spatial coordinates (x, y, z)
    x_b = path[j][:3]  # Extract only the spatial coordinates (x, y, z)

    if X.collision_free(x_a, x_b, r):
        new_path = new_path[:i + 1] + new_path[j:]  # Remove the points between i and j

    return new_path

# Adjusted find_closest_point to ensure both path points and goal are in 3D
def find_closest_point(path, goal):
    # Convert goal to 3D by appending a z-coordinate of 0 if it's 2D
    goal_3d = np.array([goal[0], goal[1], goal[2] if len(goal) == 3 else 0])
    
    distances = []
    for point in path:
        # Ensure each path point is in 3D by adding z=0 if it's 2D
        point_3d = np.array([point[0], point[1], point[2] if len(point) == 3 else 0])
        distances.append(np.linalg.norm(point_3d - goal_3d))
    
    return np.argmin(distances)


print("RRT to first intermediate goal...")
rrt = RRT(X, q, x_init_3d, x_goal_1_3d, max_samples, r, prc)
path1 = rrt.rrt_search()

if path1 is not None:
    print("RRT to second intermediate goal...")
    rrt = RRT(X, q, x_goal_1_3d, x_goal_2_3d, max_samples, r, prc)
    path2 = rrt.rrt_search()

    if path2 is not None:
        print("RRT to third intermediate goal...")
        rrt = RRT(X, q, x_goal_2_3d, x_goal_3_3d, max_samples, r, prc)
        path3 = rrt.rrt_search()

        if path3 is not None:
            print("RRT to fourth intermediate goal...")
            rrt = RRT(X, q, x_goal_3_3d, x_goal_4_3d, max_samples, r, prc)
            path4 = rrt.rrt_search()

            if path4 is not None:
                print("RRT to fifth intermediate goal...")
                rrt = RRT(X, q, x_goal_4_3d, x_goal_5_3d, max_samples, r, prc)
                path5 = rrt.rrt_search()

                if path5 is not None:
                    print("RRT to sixth intermediate goal...")
                    rrt = RRT(X, q, x_goal_5_3d, x_goal_6_3d, max_samples, r, prc)
                    path6 = rrt.rrt_search()

                    if path6 is not None:
                        print("RRT to final goal...")
                        rrt = RRT(X, q, x_goal_6_3d, x_final_goal_3d, max_samples, r, prc)
                        path_final = rrt.rrt_search()

                        if path_final is not None:
                            # Combine all paths
                            path = path1 + path2[1:] + path3[1:] + path4[1:] + path5[1:] + path6[1:] + path_final[1:]
                        else:
                            path = path1 + path2[1:] + path3[1:] + path4[1:] + path5[1:] + path6[1:]
                    else:
                        path = path1 + path2[1:] + path3[1:] + path4[1:] + path5[1:]
                else:
                    path = path1 + path2[1:] + path3[1:] + path4[1:]
            else:
                path = path1 + path2[1:] + path3[1:]
        else:
            path = path1 + path2[1:]
    else:
        path = path1
else:
    path = None

# Apply DWA for local optimization along the RRT path
# Check if a valid path was generated before proceeding
if path is not None:
    # Apply DWA for local optimization along the RRT path
    dwa = DWA(dwa_params)
    optimized_path = []

    start_time = time.time()
    for i in range(len(path) - 1):
        start_point = path[i] + (0.0, 0.0)  # Initialize v and w to 0, using tuple
        end_point = path[i + 1]

        local_path = dwa.plan(start_point, end_point, X, all_obstacles)
        optimized_path.extend(local_path)
    dwa_time = time.time() - start_time

    # Apply ALNS optimization to each segment of the path
    print("Applying ALNS optimization...")
    alns_optimized_path = []

    # Define all goals including the final goal in sequence
    goals = [x_goal_1, x_goal_2, x_goal_3, x_goal_4, x_goal_5, x_goal_6, x_final_goal]

    # Segment-based ALNS optimization over each segment from start to each intermediate and final goal
    previous_index = 0  # Initialize the starting index
    for i, goal in enumerate(goals):
    # Find the closest point in the optimized path to the current goal
       index_goal = find_closest_point(optimized_path, goal)

    # Define the segment from the last goal (or start) to the current goal
       segment = optimized_path[previous_index:index_goal + 1]
    
    # Debugging output to verify segment extraction
       print(f"Optimizing segment from goal {i} to goal {i+1}, segment length: {len(segment)}")
    
    # Apply ALNS optimization to the segment
       alns_optimized_segment = alns_optimize_path(segment, X, all_obstacles, max_iterations=100)

    # Add the ALNS optimized segment to the final optimized path
       if alns_optimized_path:
        # Avoid duplicating points between segments
           alns_optimized_path.extend(alns_optimized_segment[1:])  
       else:
           alns_optimized_path.extend(alns_optimized_segment)

    # Update the previous_index to the current goal's index
       previous_index = index_goal
    previous_index = index_goal

# Final verification
print(f"Total ALNS optimized path length: {len(alns_optimized_path)}")
# Final verification
 
    # Ensure all paths are correctly formatted and flatten nested structures
def validate_and_correct_path(path):
    corrected_path = []
    for idx, point in enumerate(path):
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            if isinstance(point[0], (list, tuple)):
                point = [item for sublist in point for item in sublist]
            corrected_path.append(point[:3])  # Ensure only the first three elements are used
        else:
            raise ValueError(f"Point {idx} in path is incorrectly formatted: {point}")
    return corrected_path
# Initialize default metrics in case paths are not generated
rrt_path_length = dwa_path_length = alns_path_length = 0
rrt_smoothness = dwa_smoothness = alns_smoothness = 0
rrt_clearance = dwa_clearance = alns_clearance = float('inf')
rrt_energy = dwa_energy = alns_energy = 0

# Check if paths are generated before calculating metrics
if path is not None:
    rrt_path_length = path_length(path)
    rrt_smoothness = path_smoothness(path)
    rrt_clearance = min_obstacle_clearance(path, all_obstacles)
    rrt_energy = compute_energy_usage(path, dwa_params['max_speed'])

if optimized_path:
    dwa_path_length = path_length(optimized_path)
    dwa_smoothness = path_smoothness(optimized_path)
    dwa_clearance = min_obstacle_clearance(optimized_path, all_obstacles)
    dwa_energy = compute_energy_usage(optimized_path, dwa_params['max_speed'])

if alns_optimized_path:
    alns_path_length = path_length(alns_optimized_path)
    alns_smoothness = path_smoothness(alns_optimized_path)
    alns_clearance = min_obstacle_clearance(alns_optimized_path, all_obstacles)
    alns_energy = compute_energy_usage(alns_optimized_path, dwa_params['max_speed'])

# Print the metrics
print(f"RRT Path Length: {rrt_path_length}")
print(f"DWA Optimized Path Length: {dwa_path_length}")
print(f"ALNS Optimized Path Length: {alns_path_length}")
print(f"RRT Path Smoothness: {rrt_smoothness}")
print(f"DWA Optimized Path Smoothness: {dwa_smoothness}")
print(f"ALNS Optimized Path Smoothness: {alns_smoothness}")
print(f"RRT Path Minimum Clearance: {rrt_clearance}")
print(f"DWA Optimized Path Minimum Clearance: {dwa_clearance}")
print(f"ALNS Optimized Path Minimum Clearance: {alns_clearance}")
print(f"RRT Energy Usage: {rrt_energy}")
print(f"DWA Energy Usage: {dwa_energy}")
print(f"ALNS Energy Usage: {alns_energy}")

ml_path = np.array([])
# Ensure the final paths are properly flattened and structured
def flatten_path_points(path):
    cleaned_path = []
    for idx, point in enumerate(path):
        if isinstance(point, np.ndarray):
            flat_point = point.flatten().tolist()
            if len(flat_point) >= 3:
                cleaned_path.append(flat_point[:3])  # Take only the first three elements (x, y, z)
            else:
                raise ValueError(f"Point {idx} in optimized_path is incorrectly formatted: {flat_point}")
        elif isinstance(point, (list, tuple)) and len(point) >= 3:
            cleaned_path.append(point[:3])  # Handle lists or tuples directly
        else:
            raise ValueError(f"Point {idx} in optimized_path is not correctly formatted: {point}")
    return np.array(cleaned_path)

# Apply the flattening function to optimized_path before plotting
try:
    optimized_path = flatten_path_points(optimized_path)
    ml_path = flatten_path_points(ml_path)
except ValueError as e:
    print(f"Error encountered during flattening: {e}")

# Plotting and Animation
# Check that all paths are available and contain data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(X_dimensions[0])
ax.set_ylim(X_dimensions[1])
ax.set_zlim(X_dimensions[2])

# Initial empty lines for each path
line_rrt, = ax.plot([], [], [], linestyle='--', color='red', label="RRT Path")
line_dwa, = ax.plot([], [], [], linestyle='--', color='blue', label="DWA Optimized Path")
line_alns, = ax.plot([], [], [], linestyle='--', color='green', label="ALNS Optimized Path")

# Scatter plot for start and goal points with labels
goals = {
    "Start": (x_init_3d, 'magenta'),
    "Goal 1": (x_goal_1_3d, 'pink'),
    "Goal 2": (x_goal_2_3d, 'cyan'),
    "Goal 3": (x_goal_3_3d, 'yellow'),
    "Goal 4": (x_goal_4_3d, 'purple'),
    "Goal 5": (x_goal_5_3d, 'orange'),
    "Goal 6": (x_goal_6_3d, 'lime'),
    "Final Goal": (x_final_goal_3d, 'green')
}

# Scatter plot for start and goal points with labels
for label, (point, color) in goals.items():
    ax.scatter(*point, color=color, label=label)
    ax.text(point[0], point[1], point[2], label, fontsize=9, ha='right')


# Plot obstacles as cylinders
for obs in all_obstacles:
    x_center, y_center, z_min, z_max, radius = obs
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(z_min, z_max, 100)
    x = radius * np.outer(np.cos(u), np.ones(len(v))) + x_center
    y = radius * np.outer(np.sin(u), np.ones(len(v))) + y_center
    z = np.outer(np.ones(len(u)), v)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

# Initialize empty data arrays for the lines
rrt_x, rrt_y, rrt_z = [], [], []
dwa_x, dwa_y, dwa_z = [], [], []
alns_x, alns_y, alns_z = [], [], []

# Animation update function
def update(num):
    # Update RRT path
    if num < len(path):
        rrt_x.append(path[num][0])
        rrt_y.append(path[num][1])
        rrt_z.append(path[num][2])
        line_rrt.set_data(rrt_x, rrt_y)
        line_rrt.set_3d_properties(rrt_z)

    # Update DWA path
    if num < len(optimized_path):
        dwa_x.append(optimized_path[num][0])
        dwa_y.append(optimized_path[num][1])
        dwa_z.append(optimized_path[num][2])
        line_dwa.set_data(dwa_x, dwa_y)
        line_dwa.set_3d_properties(dwa_z)

    # Update ALNS path
    if num < len(alns_optimized_path):
        alns_x.append(alns_optimized_path[num][0])
        alns_y.append(alns_optimized_path[num][1])
        alns_z.append(alns_optimized_path[num][2])
        line_alns.set_data(alns_x, alns_y)
        line_alns.set_3d_properties(alns_z)

    return line_rrt, line_dwa, line_alns

# Set up animation
num_frames = max(len(path), len(optimized_path), len(alns_optimized_path))
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

# Show plot with legend
ax.legend()
plt.show()



