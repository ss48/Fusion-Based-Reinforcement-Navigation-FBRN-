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
from rrt_algorithms.search_space.search_space3 import SearchSpace
from rrt_algorithms.utilities.plotting2 import Plot
from rrt_algorithms.dwa_algorithm.DWA4 import DWA, FuzzyController
from rrt_algorithms.dqn_algorithm.DQN0 import DQNAgent
from rrt_algorithms.dqn_algorithm.DQN0 import DroneEnv
from rrt_algorithms.utilities.obstacle_generation3 import generate_random_cylindrical_obstacles 
import time  # Import for time tracking

# Initialize cube times and add placeholders for text annotations
cube_times = {
    'cube1': {'start': time.time(), 'end': None},
    'cube2': {'start': time.time(), 'end': None},
    'cube3': {'start': time.time(), 'end': None}
}


# Define the search space dimensions
X_dimensions = np.array([(-200, 200), (-200, 200), (-150, 150)])


# Initial and goal positions
x_init = (150, 50, -82)
x_intermediate = (-180, -150, -150)
x_second_goal = (200, 160, 30)
x_third_goal = (-120, 160, 10)
x_final_goal = (150, -100, -50)

# Generate random obstacles
n = 4 # Number of random obstacles


# Define static obstacles
obstacles = [
    (-50, 160, -150, 285, 35),  #(center coordinate, height,diameter)
    (-80, 20, 0, 160, 39),   
    (70, 30, 0, 190, 25),   
    (-80, -180, 50, 170, 22), 
    (0, -50, -30, 80, 32),
]

cube_size = 5
#cube_position = np.array(x_init)

# Set the initial positions for each cube (use goal positions or starting positions as required)
cube1_position = np.array(x_init)           # Initial position for Cube 1
cube2_position = np.array(x_intermediate)    # Initial position for Cube 2
cube3_position = np.array(x_second_goal)     # Initial position for Cube 3
final_goal = np.array(x_final_goal)          # Shared final goal for all cubes





# Store all initial positions and assign each one to a cube's start position

# Define paths for each cube from their start position to the final goal
 
    
# Function to create vertices for the cube
def create_cube_vertices(center, size):
    d = size / 2.0
    return np.array([
        center + np.array([-d, -d, -d]),
        center + np.array([-d, -d,  d]),
        center + np.array([-d,  d, -d]),
        center + np.array([-d,  d,  d]),
        center + np.array([ d, -d, -d]),
        center + np.array([ d, -d,  d]),
        center + np.array([ d,  d, -d]),
        center + np.array([ d,  d,  d]),
    ])

# Plot each cube with the initial vertices
def plot_cube(ax, vertices, color='orange'):
    pairs = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    lines = []
    for p1, p2 in pairs:
        line, = ax.plot([vertices[p1][0], vertices[p2][0]],
                        [vertices[p1][1], vertices[p2][1]],
                        [vertices[p1][2], vertices[p2][2]],
                        color=color)
        lines.append(line)
    return lines
    

# Combine static and randomly generated obstacles into one list
X = SearchSpace(X_dimensions)
# Initialize the SearchSpace with all obstacles
Obstacles = generate_random_cylindrical_obstacles(X, x_init, x_final_goal, n)

all_obstacles = obstacles + Obstacles
X = SearchSpace(X_dimensions, all_obstacles)
# RRT parameters
q = 300
r = 1
max_samples = 5024
prc = 0.1
def add_vertex(self, tree, v):
    if isinstance(v, tuple) and len(v) == 3:
        v_composite = (v[0], v[1], v[2], v[2])  # Adding z as a 2D bounding pair
    else:
        v_composite = v
    self.trees[tree].V.insert(0, v_composite, v)
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
def generate_collision_free_rrt(start, goal, search_space, obstacles, max_attempts=20):
    for _ in range(max_attempts):
        rrt = RRT(search_space, q=30, x_init=start, x_goal=goal, max_samples=2024, r=1, prc=0.1)
        path = rrt.rrt_search()
        
        # Enhanced obstacle avoidance: ensure each segment avoids obstacles
        if path:
            path_is_safe = True
            for i in range(len(path) - 1):
                if not search_space.collision_free(path[i], path[i+1], steps=50):  # Increase granularity of checks
                    path_is_safe = False
                    break
            if path_is_safe:
                return path  # Return only if entire path is obstacle-free
    return None


def generate_full_rrt_path(start, goals, search_space, obstacles, max_attempts=10):
    complete_path = []
    current_start = start

    for goal in goals:
        path_segment = generate_collision_free_rrt(current_start, goal, search_space, obstacles, max_attempts)
        if path_segment is None:
            print(f"Failed to find path to goal: {goal}")
            return None
        complete_path.extend(path_segment if not complete_path else path_segment[1:])
        current_start = goal

    return complete_path
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
def alns_optimize_path(path, X, all_obstacles, max_iterations=100):
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

def find_closest_point(path, goal):
    distances = [np.linalg.norm(np.array(p[:3]) - np.array(goal)) for p in path]
    return np.argmin(distances)

# RRT pathfinding to the first intermediate goal
print("RRT to first intermediate goal...")
start_time = time.time()
rrt = RRT(X, q, x_init, x_intermediate, max_samples, r, prc)
path1 = rrt.rrt_search()
rrt_time = time.time() - start_time

if path1 is not None:
    print("RRT to 1st to 2nd goal...")
    rrt = RRT(X, q, x_intermediate, x_second_goal, max_samples, r, prc)
    path2 = rrt.rrt_search()

    if path2 is not None:
        print("RRT to 2nd to 3rd goal...")
        rrt = RRT(X, q, x_second_goal, x_third_goal, max_samples, r, prc)
        path3 = rrt.rrt_search()
            
        if path3 is not None:
            print("RRT to final goal...")
            rrt = RRT(X, q, x_third_goal, x_final_goal, max_samples, r, prc)
            path_final = rrt.rrt_search()

        # Combine all paths
            if path_final is not None:
                path = path1 + path2[1:] + path3[1:] + path_final[1:] 
            else:
                path = path1 + path2[1:] + path3[1:] 
        else:
            path = path1
    else:
        path = None




# Apply DWA for local optimization along the RRT path
if path is not None:
    dwa = DWA(dwa_params)
    optimized_path = []

    start_time = time.time()
    for i in range(len(path) - 1):
        start_point = path[i] + (0.0, 0.0)  # Initialize v and w to 0, using tuple
        end_point = path[i + 1]

        # Enhanced DWA path planning with high-resolution collision checking
        local_path = dwa.plan(start_point, end_point, X, all_obstacles)  # Increased steps for finer checks
        optimized_path.extend(local_path)
    dwa_time = time.time() - start_time

    # Apply ALNS optimization to each segment of the path
    print("Applying ALNS optimization...")
    alns_optimized_path = []
    initial_positions = [cube1_position, cube2_position, cube3_position]



    # Find the closest points to the goals
    index_intermediate = find_closest_point(optimized_path, x_intermediate)
    index_second_goal = find_closest_point(optimized_path, x_second_goal)
    index_third_goal = find_closest_point(optimized_path, x_third_goal)
    # Segment 1: Start to Intermediate
    segment1 = optimized_path[:index_intermediate + 1]
    alns_optimized_segment1 = alns_optimize_path(segment1, X, all_obstacles, max_iterations=100)
    
    # Segment 2: Intermediate to Second Goal
    segment2 = optimized_path[index_intermediate:index_second_goal + 1]
    alns_optimized_segment2 = alns_optimize_path(segment2, X, all_obstacles, max_iterations=100)
    
    # Segment 3: Second Goal to Final Goal
    segment3 = optimized_path[index_second_goal:]
    alns_optimized_segment3 = alns_optimize_path(segment3, X, all_obstacles, max_iterations=100)
    
        # Segment 3: Second Goal to Final Goal
    segment4 = optimized_path[index_third_goal:]
    alns_optimized_segment4 = alns_optimize_path(segment4, X, all_obstacles, max_iterations=100)
   
    
    # Combine the ALNS-optimized segments
    alns_optimized_path = alns_optimized_segment1 + alns_optimized_segment2[1:] + alns_optimized_segment3[1:] + alns_optimized_segment4 + alns_optimized_segment1

    # Flatten nested structures if necessary
    alns_optimized_path = [point[:3] for point in alns_optimized_path]  # Keep only (x, y, z) coordinates

    goals = [x_intermediate, x_second_goal, x_third_goal, x_final_goal]  # Adjust based on desired goals
    
    paths = []
    
    initial_positions = [cube1_position, cube2_position, cube3_position]
    goals = [x_intermediate, x_second_goal, x_third_goal, x_final_goal]

    for start_pos in initial_positions:
        # Generate the full path for each cube with ALNS applied
        path_segment = generate_full_rrt_path(start_pos, goals, X, all_obstacles, max_attempts=20)
    
        if path_segment:
            # Optimize the path using ALNS
            optimized_path_segment = alns_optimize_path(path_segment, X, all_obstacles, max_iterations=100)
            paths.append(optimized_path_segment)
        else:
            print(f"Warning: Failed to find a path from {start_pos} to final goal.")




    # Ensure all paths are correctly formatted and flatten nested structures

def validate_and_correct_path(path, label="Path"):
    """
    Ensures that the path only contains (x, y, z) points and handles nested structures.
    Returns a cleaned path if valid; raises a warning and skips if not.
    """
    corrected_path = []
    for idx, point in enumerate(path):
        if isinstance(point, np.ndarray):
            flat_point = point.flatten().tolist()
        elif isinstance(point, (list, tuple)):
            flat_point = point
        else:
            print(f"Warning: {label} at index {idx} is not an array or tuple. Skipping.")
            continue
        
        # Ensure (x, y, z) format and check for NaN values
        if len(flat_point) >= 3 and not any(np.isnan(flat_point[:3])):
            corrected_path.append(flat_point[:3])
        else:
            print(f"Warning: {label} at index {idx} is incorrectly formatted or contains NaNs. Skipping.")
    return np.array(corrected_path) if corrected_path else None
env = DroneEnv(X, x_init, x_final_goal, all_obstacles, dwa_params)
# --- Generate ML Path with Validation ---
# Assuming the environment `env` and the DQN agent are defined elsewhere in the code
ml_path = []
state = env.reset()
done = False

while not done:
    action = env.agent.act(state)
    next_state, _, done = env.step(action)
    ml_path.append(next_state[0][:3])  # Only keep (x, y, z) coordinates
    state = next_state

# Validate the ML path after generation
ml_path = validate_and_correct_path(ml_path, label="ML Path")

if ml_path is not None:
    print("ML path successfully generated and validated.")
else:
    print("Warning: ML path could not be validated and will not be plotted.")



    # Validate and correct paths
optimized_path = validate_and_correct_path(optimized_path)
alns_optimized_path = validate_and_correct_path(alns_optimized_path)

# Metrics calculation
# Initialize default metrics in case paths are not generated
rrt_path_length = dwa_path_length = alns_path_length = 0
rrt_smoothness = dwa_smoothness = alns_smoothness = 0
rrt_clearance = dwa_clearance = alns_clearance = float('inf')
rrt_energy = dwa_energy = alns_energy = 0

# Check if paths are generated before calculating metrics
# Check if paths are generated before calculating metrics
if path is not None:
    rrt_path_length = path_length(path)
    rrt_smoothness = path_smoothness(path)
    rrt_clearance = min_obstacle_clearance(path, all_obstacles)
    rrt_energy = compute_energy_usage(path, dwa_params['max_speed'])

if optimized_path is not None and optimized_path.size > 0:
    dwa_path_length = path_length(optimized_path)
    dwa_smoothness = path_smoothness(optimized_path)
    dwa_clearance = min_obstacle_clearance(optimized_path, all_obstacles)
    dwa_energy = compute_energy_usage(optimized_path, dwa_params['max_speed'])

if alns_optimized_path is not None and alns_optimized_path.size > 0:
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

average_velocity = dwa_params['max_speed']
rrt_energy = compute_energy_usage(path, average_velocity)
dwa_energy = compute_energy_usage(optimized_path, average_velocity)
alns_energy = compute_energy_usage(alns_optimized_path, average_velocity)

# Assuming each cube starts moving at different points in the script
if cube_times['cube1']['start'] is None:
    cube_times['cube1']['start'] = time.time()  # Record start time for Cube 1

if cube_times['cube2']['start'] is None:
    cube_times['cube2']['start'] = time.time()  # Record start time for Cube 2

if cube_times['cube3']['start'] is None:
    cube_times['cube3']['start'] = time.time()  # Record start time for Cube 3

# Training the DQN

batch_size = 32
# Generate the ML path using the trained model

# Train the DQN model
#for episode in range(num_episodes):
#    state = env.reset()
#    done = False
#    total_reward = 0

#    while not done:
#        action = env.agent.act(state)
#        next_state, reward, done = env.step(action)
#        env.agent.remember(state, action, reward, next_state, done)
#        state = next_state
#        total_reward += reward

 #       if len(env.agent.memory) > batch_size:
 #           env.agent.replay(batch_size)

 #   print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {env.agent.epsilon:.4f}")

# Save the trained model
env.agent.model.save("dqn_model_optimized.keras")
print("Optimized DQN model saved as dqn_model_optimized.keras")



while not done:
    action = env.agent.act(state)  # Use the trained policy to act
    next_state, _, done = env.step(action)
    ml_path.append(next_state)
    state = next_state

# Convert ml_path to a numpy array for easy handling
ml_path = np.array(ml_path)

# Check if ml_path has valid data
if ml_path.size == 0 or len(ml_path) < 3:
    print("Warning: ML path is not available or has insufficient points.")
else:
    print("ML path successfully generated.")


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



# First figure: RRT, DWA, ALNS paths with ML path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(type(ax))  
ax.set_xlim(X_dimensions[0])
ax.set_ylim(X_dimensions[1])
ax.set_zlim(X_dimensions[2])

# Initialize timer text objects for each cube
#cube1_timer_text = ax.text2D(0.05, 0.95, "Cube 1 Time: 0.00s", transform=ax.transAxes, fontsize=10, color='orange')
#cube2_timer_text = ax.text2D(0.05, 0.90, "Cube 2 Time: 0.00s", transform=ax.transAxes, fontsize=10, color='blue')
#cube3_timer_text = ax.text2D(0.05, 0.85, "Cube 3 Time: 0.00s", transform=ax.transAxes, fontsize=10, color='green')
# Initialize time display annotations
time_texts = [
    ax.text2D(0.05, 0.9 - i * 0.05, "", transform=ax.transAxes)
    for i in range(3)
]



goals = {
    "Start": x_init,
    "2nd Goal": x_intermediate,
    "3rd Goal": x_second_goal,
    "4th Goal": x_third_goal,
    "Final Goal": x_final_goal
}

for label, coord in goals.items():
    ax.scatter(*coord, label=label, s=50)
    ax.text(coord[0], coord[1], coord[2], label, fontsize=9, ha='right')

# Generate optimized path (Assuming `alns_optimized_path` exists and is a list of points)
# This should be calculated before this section
alns_optimized_path = [x_init, x_intermediate, x_second_goal, x_third_goal, x_final_goal]  # Placeholder for testing



# Initial position for the moving cube
cube_vertices = create_cube_vertices(cube1_position, cube_size)
#cube = plot_cube(ax, cube_vertices)

# Set up the speed of the cube
speed = 0.02 # distance per frame

#the following lines draw three stationary cubes at6 the starting point 
#cube1 = plot_cube(ax, create_cube_vertices(cube1_position, cube_size), color='orange')
#cube2 = plot_cube(ax, create_cube_vertices(cube2_position, cube_size), color='blue')
#cube3 = plot_cube(ax, create_cube_vertices(cube3_position, cube_size), color='green')

ax.set_xlabel('X Axis (m)')
ax.set_ylabel('Y Axis (m)')
ax.set_zlabel('Z Axis (m)')



# Ensure the ALNS, RRT, and DWA paths have valid coordinates
if path is not None and len(path) > 1:
    ax.plot(*zip(*path), linestyle='--', color='red', label="RRT Path")
else:
    print("Warning: RRT path is not available or has insufficient points.")

if optimized_path is not None and len(optimized_path) > 1:
    ax.plot(*zip(*optimized_path), linestyle='--', color='blue', label="DWA Optimized Path")
else:
    print("Warning: DWA optimized path is not available or has insufficient points.")

if alns_optimized_path is not None and len(alns_optimized_path) > 1:
    ax.plot(*zip(*alns_optimized_path), linestyle='--', color='green', label="ALNS Optimized Path")
else:
    print("Warning: ALNS optimized path is not available or has insufficient points.")

# Ensure mlpath has valid coordinates if it is plotted
#if ml_path.size > 0 and len(ml_path) >= 2:
#    ax.plot(ml_path[:, 0], ml_path[:, 1], ml_path[:, 2], color='black', label="ML Path")
#    print(f"ml_path length: {len(ml_path)}")
#    print(f"ml_path data:\n{ml_path}")
#else:
#    print("Warning: ML path is not available or has insufficient points.")



# Plot paths if they are validated
if ml_path is not None:
    ax.plot(ml_path[:, 0], ml_path[:, 1], ml_path[:, 2], color='black', label="ML Path")




# Plot static obstacles
for obs in all_obstacles:
    x_center, y_center, z_min, z_max, radius = obs
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(z_min, z_max, 100)
    x = radius * np.outer(np.cos(u), np.ones(len(v))) + x_center
    y = radius * np.outer(np.sin(u), np.ones(len(v))) + y_center
    z = np.outer(np.ones(len(u)), v)
    ax.plot_surface(x, y, z, color='red', alpha=0.3)

# Plot start and goal points
#ax.scatter(*x_init, color='magenta', label="Start")
#ax.scatter(*x_intermediate, color='pink', label="Intermediate Goal")
#ax.scatter(*x_second_goal, color='cyan', label="Second Goal")
#ax.scatter(*x_final_goal, color='green', label="Final Goal")

# Animation of the ML path on the same figure
drone_path_line, = ax.plot([], [], [], color='orange', linewidth=2)


# Animation update function
distance_covered1 = 0.0
distance_covered2 = 0.0
distance_covered3 = 0.0

# Initial positions of the cubes
cube_start_positions = [alns_optimized_path[0], alns_optimized_path[0], alns_optimized_path[0]]  # All start at the beginning of the ALNS path

# Plot each cube at its initial position

# Distances covered by each cube
distances_covered = [0.0, 0.0, 0.0]
# Speed of movement along the path
#speeds = [2.0, 1.2, 2.3]  # Different speeds for each cube

cube_plots = [
    plot_cube(ax, create_cube_vertices(cube_start_positions[i], cube_size), color=color)
    for i, color in enumerate(['blue','orange', 'green'])
]

# Function to get position along ALNS path
# Check that cube_positions (ALNS path) is not empty
if not alns_optimized_path:
    print("Error: ALNS optimized path is empty. Ensure ALNS path is correctly computed.")
else:
    # Proceed if alns_optimized_path has valid points
    cube_positions = [alns_optimized_path[0:], alns_optimized_path[1:], alns_optimized_path[2:]]

    # Function to get position along ALNS path with validation
# Helper function to find the interpolated position on the path based on distance
def get_position_along_path(path, distance):
    # Calculate the cumulative distances along the path
    cumulative_distances = [0]
    for i in range(1, len(path)):
        cumulative_distances.append(
            cumulative_distances[i-1] + np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        )

    # Find the two points between which the current distance falls
    for i in range(1, len(cumulative_distances)):
        if cumulative_distances[i] >= distance:
            # Interpolate between points i-1 and i
            segment_distance = cumulative_distances[i] - cumulative_distances[i-1]
            t = (distance - cumulative_distances[i-1]) / segment_distance  # Interpolation factor
            interpolated_position = (1 - t) * np.array(path[i-1]) + t * np.array(path[i])
            return interpolated_position
    return path[-1]  # Return the last point if distance exceeds path length


# Define a safe distance threshold to begin offsetting Cube 1
SAFE_DISTANCE = 25.0  # Safe distance threshold between cubes


# Define a small threshold for goal distance
GOAL_THRESHOLD = 1.0  # Adjust as needed based on environment scale

# Update `update_cubes` to capture end time when each cube reaches the goal
#collision_radius = np.sqrt(3 * (cube_size / 2) ** 2)
collision_radius = cube_size * np.sqrt(3) * 5 / 2  # Radius for cube-to-cube collision detection


# Define `offset_applied` globally if it should persist across frames
offset_applied = [False] * len(cube_plots)  # Initialize offset tracking for each cube

# Enhanced function to check collision-free paths that includes the cube's collision radius
def collision_free_expanded(X, start, end, radius):
    """
    Checks if the path from start to end is free of obstacles, considering the specified radius around the path.
    """
    # Validate that start and end are defined and contain no NaN values
    if np.any(np.isnan(start)) or np.any(np.isnan(end)):
        print("Warning: NaN values detected in start or end positions. Skipping collision check.")
        return False

    # Compute the step count safely
    step_vector = np.array(end) - np.array(start)
    distance = np.linalg.norm(step_vector)
    
    # Check for zero distance to avoid division by zero or NaN issues
    if distance == 0:
        return True

    # Determine the number of steps to check along the path
    steps = int(distance / (radius / 2)) + 1  # Increase granularity
    for step in range(steps + 1):
        position = np.array(start) + step * step_vector / steps
        if not X.collision_free(position - radius, position + radius):
            return False
    return True


# Flag and start time for return delay
waiting_at_goal = [False] * len(cube_plots)  # Track if each cube is waiting at the goal
return_start_time = [None] * len(cube_plots)  # Start time for the return wait


# Define global safe distance and offset distance for the return trip
return_safe_distance = 30.0  # Safe distance when returning
return_offset_distance = 25.0  # Offset distance for collision avoidance on the return path

# Initialize text annotations for speed and distance for each cube
speed_texts = [
    ax.text2D(0.15, 0.9 - i * 0.05, "", transform=ax.transAxes)
    for i in range(3)
]
distance_texts = [
    ax.text2D(0.25, 0.9 - i * 0.05, "", transform=ax.transAxes)
    for i in range(3)
]




# Create the nibble (direction indicator) for each cube
nibble_plots = []
for color in ['blue', 'orange', 'green']:
    nibble_line, = ax.plot([], [], [], color=color, linewidth=2)  # Creates the nibble line
    nibble_plots.append(nibble_line)

cube_speeds = [random.uniform(1.5, 5.5) for _ in range(len(cube_plots))]
cube_yaws = [random.uniform(-40,40) for _ in range(len(cube_plots))]


#cube_speeds = [random.uniform(dwa_params['min_speeds'], dwa_params['max_speed']) for _ in range(len(cube_plots))]
#cube_yaws = [random.uniform(-dwa_params['max_yaw_rate'], dwa_params['max_yaw_rate']) for _ in range(len(cube_plots))]
cube_priorities = [0] * len(cube_plots)  # Start with equal priority

# Dynamically update cube priorities based on their speed




nib_length = 5.0

# Create an array to store the nib plot elements
nib_plots = [ax.plot([], [], [], color=color, linewidth=2)[0] for color in ['black', 'black', 'black']]

# Define dynamic priority update function
def update_priority(cube_speeds):
    """
    Dynamically updates priorities based on cube speeds and other conditions.
    Higher speed gets higher priority, for example.
    """
    sorted_indices = np.argsort(cube_speeds)[::-1]  # Sort by speed descending
    new_priorities = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
    return new_priorities

yaw_angles = [0.0] * len(cube_plots)  # Start with 0 yaw for all cubes

# Length of nibble to indicate cube direction
nibble_length = 5.0  # Adjust as needed

def update_yaw_and_direction(cube_index, new_position, previous_position):
    """
    Updates the yaw angle based on the direction of movement along the path.
    Returns the yaw angle in radians.
    """
    # Calculate the direction vector along the path
    direction_vector = new_position[:2] - previous_position[:2]  # Ignore z-axis for 2D yaw calculation
    if np.linalg.norm(direction_vector) > 0:
        yaw = np.arctan2(direction_vector[1], direction_vector[0])  # Calculate yaw angle in radians
    else:
        yaw = yaw_angles[cube_index]  # Maintain previous yaw if direction vector is zero

    yaw_angles[cube_index] = yaw  # Update the yaw angle
    return yaw


def calculate_nib_position(cube_index, position):
    """
    Calculates the nib position based on yaw angle for the specified cube.
    """
    yaw = yaw_angles[cube_index]
    forward_direction = np.array([np.cos(yaw), np.sin(yaw), 0])  # Forward vector based on yaw
    nib_start = position
    nib_end = position + nibble_length * forward_direction  # Extend in forward direction
    return nib_start, nib_end

SAFE_DISTANCE = 20.0  # Minimum safe distance between cubes
SLOW_DOWN_DISTANCE = 25.0  # Distance at which cubes start slowing down

AVOIDANCE_FACTOR = 8.0  # Offset distance when avoiding

# Adjust constants for avoidance behavior


STOP_DISTANCE = 10.0         # Distance at which cubes temporarily stop to avoid collision
MAX_PAUSE_DURATION = 2.0    # Maximum duration in seconds for which a cube can pause
ORIGINAL_SPEEDS = cube_speeds.copy()  # Store original speeds for resetting after avoidance

# Function to adjust cube speeds dynamically based on proximity to the goal and other cubes
def adjust_speed(i, new_position, distances_covered, path):
    # Adjust speed near the goal
    goal_distance = np.linalg.norm(new_position[:3] - np.array(path[-1]))
    speed = ORIGINAL_SPEEDS[i] * 0.3 if goal_distance < 20 else ORIGINAL_SPEEDS[i]

    # Reduce speed if another cube is nearby
    for j, other_path in enumerate(cube_positions):
        if i != j:
            other_position = np.array(get_position_along_path(other_path, distances_covered[j]))
            if np.linalg.norm(new_position - other_position) < SLOW_DOWN_DISTANCE:
                speed *= 0.5  # Slow down if too close to another cube

    return speed

# Modify collision avoidance logic to prevent indefinite pausing
def collision_avoidance(i, new_position, priorities, distances_covered):
    offset_applied = False  # Track if any offset was applied
    for j, other_path in enumerate(cube_positions):
        if i != j:
            other_position = np.array(get_position_along_path(other_path, distances_covered[j]), dtype=np.float64)
            distance_to_other = np.linalg.norm(new_position - other_position)

            # Adjust speed based on proximity
            if distance_to_other < SAFE_DISTANCE:
                # Apply lateral offset instead of stopping
                direction_vector = new_position - other_position
                perpendicular_offset = np.cross(direction_vector, [0, 0, 1])
                if np.linalg.norm(perpendicular_offset) > 0:
                    perpendicular_offset = (SAFE_DISTANCE / np.linalg.norm(perpendicular_offset)) * perpendicular_offset
                    new_position += perpendicular_offset
                    offset_applied = True
                    print(f"Cube {i} applied lateral offset to avoid Cube {j}.")
    
    # Ensure at least some speed is maintained to avoid deadlock
    if not offset_applied:
        cube_speeds[i] = max(cube_speeds[i] - PAUSE_SPEED_REDUCTION_RATE, 0.05)
    else:
        cube_speeds[i] = original_speeds[i]
    
    return offset_applied

# Define constants for the gradual pause mechanism
SAFE_DISTANCE = 15.0  # Minimum safe distance between cubes
SLOW_DOWN_DISTANCE = 25.0  # Distance at which cubes start slowing down
STOP_DISTANCE = 10.0  # Distance at which cubes stop
PAUSE_SPEED_REDUCTION_RATE = 0.1  # Rate at which speed decreases
RECOVERY_SPEED_INCREMENT = 0.1  # Rate at which speed recovers after the conflict
MAX_AVOIDANCE_ATTEMPTS = 5  # Number of repeated avoidance attempts before pausing
PAUSE_DURATION = 2.0  # Duration in seconds for a temporary pause
RECOVERY_OFFSET_FACTOR = 1.2  # Slight offset adjustment factor for recovery

# Add a dictionary to track pause status and timers for each cube
pause_status = [False] * len(cube_plots)
pause_timer = [0.0] * len(cube_plots)  # Stores the timestamp when each cube started pausing
avoidance_attempts = [0] * len(cube_plots)  # Track repeated avoidance attempts
# Constants for managing the cube's behavior

OFFSET_DISTANCE = 15.0  # Offset distance for parallel path detour
GOAL_THRESHOLD = 1.0  # Threshold distance for reaching the goal
RETURN_PAUSE_DURATION = 1  # Pause duration at the goal before returning
PARALLEL_OFFSET_FACTOR = 1.0  # Factor to control offset scaling

# Flags and tracking variables for each cube
is_returning = [False] * len(cube_plots)  # Track if each cube is returning to start
temporary_offset = [False] * len(cube_plots)  # Track if each cube is in a temporary offset path
paused_at_goal = [False] * len(cube_plots)  # Track if each cube is paused at the goal
return_start_time = [None] * len(cube_plots)  # Track pause time at goal before return
AVOIDANCE_OFFSET_DISTANCE = 15.0  # Offset distance for collision avoidance

# Variables to track original speeds
original_speeds = cube_speeds.copy()  # Store each cube's original speed


def update_cubes(num):
    global distances_covered, cube_times, waiting_at_goal, return_start_time, cube_speeds, pause_status
    min_safe_distance = 30.0  # Minimum distance between cubes
    current_time = time.time()

    priorities = update_priority(cube_speeds)  # Higher speeds get higher priority

    for i, (cube, path, speed) in enumerate(zip(cube_plots, cube_positions, cube_speeds)):
        
        # Handle paused cubes
        if pause_status[i]:
            if any(
                np.linalg.norm(np.array(get_position_along_path(cube_positions[j], distances_covered[j])) - 
                               np.array(get_position_along_path(path, distances_covered[i]))) > SAFE_DISTANCE
                for j in range(len(cube_positions)) if j != i
                  ):
                # Resume movement when other cubes have moved away
                pause_status[i] = False
                cube_speeds[i] = original_speeds[i]
                print(f"Cube {i} resuming movement after pause.")
            else:
                # Skip updating this cube if it's in a paused state
                continue

        # Get new and previous positions
        new_position = np.array(get_position_along_path(path, distances_covered[i]), dtype=np.float64)
        previous_position = np.array(get_position_along_path(path, distances_covered[i] - speed))
        
        # Calculate actual distance moved in this step
        distance_step = np.linalg.norm(new_position - previous_position)
        
        # Update the cumulative distance covered
        distances_covered[i] += distance_step  # Accumulate total distance traveled

        current_speed = adjust_speed(i, new_position, distances_covered, path)
        
        if not is_returning[i] and not paused_at_goal[i]:  # Moving towards the goal
            distance_to_goal = np.linalg.norm(np.array(path[-1]) - np.array(get_position_along_path(path, distances_covered[i])))
            
            if distance_to_goal < GOAL_THRESHOLD:
                paused_at_goal[i] = True
                return_start_time[i] = current_time  # Start countdown to return
                print(f"Cube {i} reached the goal, pausing before return.")

        elif paused_at_goal[i]:  # Paused at goal, preparing for return
            # Check if pause duration is met to start returning
            if current_time - return_start_time[i] >= RETURN_PAUSE_DURATION:
                paused_at_goal[i] = False
                is_returning[i] = True
                distances_covered[i] = 0  # Reset distance for return journey
                cube_positions[i] = path[::-1]  # Reverse path for return
                print(f"Cube {i} starting return journey to start position.")

        # Apply collision avoidance with gradual pausing
        offset_applied = False  # Flag to check if offset was applied this step
        for j, other_path in enumerate(cube_positions):
            if i != j:
                other_position = np.array(get_position_along_path(other_path, distances_covered[j]), dtype=np.float64)
                distance_to_other = np.linalg.norm(new_position - other_position)

                # Collision avoidance logic
                if distance_to_other < STOP_DISTANCE:
                    if priorities[i] < priorities[j]:  # Cube `i` has lower priority
                        # Stop the cube immediately
                        cube_speeds[i] = 0.0
                        pause_status[i] = True
                        print(f"Cube {i} paused to avoid collision with Cube {j}.")
                        break
                    else:
                        # Higher-priority cube continues moving
                        cube_speeds[i] = original_speeds[i]
                
                elif distance_to_other < SLOW_DOWN_DISTANCE:
                    if priorities[i] < priorities[j]:  # Cube `i` has lower priority than Cube `j`
                        # Gradually reduce the speed of the lower-priority cube
                        cube_speeds[i] = max(cube_speeds[i] - PAUSE_SPEED_REDUCTION_RATE, 0.05)
                        print(f"Cube {i} gradually slowing down to avoid Cube {j}")
                    elif priorities[i] > priorities[j]:  # Cube `i` has higher priority
                        # Keep the higher-priority cube moving at its normal speed
                        cube_speeds[i] = original_speeds[i]

                # Apply lateral offset if within avoidance range but still moving
                if distance_to_other < SAFE_DISTANCE and not offset_applied:
                    # Offset laterally to avoid collision
                    direction_vector = new_position - other_position
                    perpendicular_offset = np.cross(direction_vector, [0, 0, 1])  # Perpendicular in XY plane
                    if np.linalg.norm(perpendicular_offset) > 0:
                        perpendicular_offset = AVOIDANCE_OFFSET_DISTANCE * (perpendicular_offset / np.linalg.norm(perpendicular_offset))
                        new_position += perpendicular_offset
                        offset_applied = True
                        print(f"Cube {i} applied lateral offset to avoid Cube {j}.")

        # Update cube's position based on its (possibly adjusted) speed
        if not pause_status[i]:  # Only update if not paused
            print(f"Cube {i} new position: {new_position}")

        # Check for NaN in position
        if np.any(np.isnan(new_position)):
            print(f"Warning: NaN detected in position for Cube {i}. Skipping update.")
            continue

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(new_position[:3] - np.array(path[-1]))
        cube_name = f'cube{i+1}'
        
        # Only mark goal reached if it's the last segment, and if within a small threshold of the final goal
        if distance_to_goal < GOAL_THRESHOLD and not waiting_at_goal[i]:
            if cube_times[cube_name]['end'] is None:
                cube_times[cube_name]['end'] = current_time
            waiting_at_goal[i] = True
            return_start_time[i] = current_time
            print(f"Cube {i} reached the goal. Waiting at goal.")
            continue
        
        # Display time and cumulative distance for the cube
        elapsed_time = (
            cube_times[cube_name]['end'] - cube_times[cube_name]['start']
            if cube_times[cube_name]['end'] is not None
            else time.time() - cube_times[cube_name]['start']
        )

        time_texts[i].set_text(f"Cube {i + 1}-Time:{elapsed_time:.2f}s;")
        speed_texts[i].set_text(f"Cube {i + 1}-Speed:{cube_speeds[i]:.2f}m/s; ")
        distance_texts[i].set_text(f"Cube {i + 1}-Distance Traveled: {distances_covered[i]:.2f}m")

        # Update yaw for the current movement direction
        yaw = update_yaw_and_direction(i, new_position, previous_position)

        # Calculate nib position based on the updated yaw
        nib_start, nib_end = calculate_nib_position(i, new_position)

        # Update nib plot for this cube
        nib_plots[i].set_data([nib_start[0], nib_end[0]], [nib_start[1], nib_end[1]])
        nib_plots[i].set_3d_properties([nib_start[2], nib_end[2]])

        # Update cube vertices to reflect the new position
        cube_vertices = create_cube_vertices(new_position, cube_size)
        for line, (p1, p2) in zip(cube, [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
        ]):
            line.set_data([cube_vertices[p1][0], cube_vertices[p2][0]],
                          [cube_vertices[p1][1], cube_vertices[p2][1]])
            line.set_3d_properties([cube_vertices[p1][2], cube_vertices[p2][2]])

    return [line for cube in cube_plots for line in cube] + time_texts + speed_texts + distance_texts + nib_plots





# --- Start Animation with Validated Paths ---
if paths:
    longest_path_length = max(len(path) for path in paths if path is not None)
    ani = animation.FuncAnimation(fig, update_cubes, frames=longest_path_length, interval=100, blit=True)
else:
    print("Error: No valid paths generated. Exiting animation setup.")


# Add legend and show plot
ax.legend()
plt.show()


