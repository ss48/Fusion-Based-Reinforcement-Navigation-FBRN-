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
X_dimensions = np.array([(-200, 200), (-200, 200), (-200, 200)])


# Initial and goal positions
x_init = (150, 50, -102)
x_intermediate = (-50, -150, -150)
x_second_goal = (200, 160, 100)
x_third_goal = (-200, 160, 100)
x_final_goal = (0, -200, 70)

# Generate random obstacles
n = 10  # Number of random obstacles


# Define static obstacles
obstacles = [
    (80, 52, -100, 85, 35),  #(center coordinate, height,diameter)
    (-80, 20, 0, 60, 59),   
    (200, 30, 0, 90, 25),   
    (-100, -180, 50, 70, 22), 
    (0, -50, -100, 80, 32),
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
Obstacles = generate_random_cylindrical_obstacles(X, (-100,-100,100), (100,100,100), n)

all_obstacles = obstacles + Obstacles
X = SearchSpace(X_dimensions, all_obstacles)
# RRT parameters
q = 30
r = 1
max_samples = 2024
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
def generate_collision_free_rrt(start, goal, search_space, obstacles, max_attempts=10):
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
    for start_pos in initial_positions:
    # Generate the full path for each cube, from start to final goal, with ALNS applied
        path_segment = generate_full_rrt_path(start_pos, goals, X, all_obstacles, max_attempts=20)
    
        if path_segment is None:
            print(f"Failed to find a path from {start_pos} to final goal")
        else:
        # Optimize the path using ALNS
            optimized_path_segment = alns_optimize_path(path_segment, X, all_obstacles, max_iterations=100)
            paths.append(optimized_path_segment)




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
# Initialize the environment
env = DroneEnv(X, x_init, x_final_goal, all_obstacles, dwa_params)

# Training parameters
num_episodes = 20
batch_size = 24

# Train the DQN model
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.agent.act(state)
        next_state, reward, done = env.step(action)
        env.agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(env.agent.memory) > batch_size:
            env.agent.replay(batch_size)

    print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {env.agent.epsilon:.4f}")

# Save the trained model
env.agent.model.save("dqn_model_optimized.keras")
print("Optimized DQN model saved as dqn_model_optimized.keras")

# Generate the ML path using the trained model
state = env.reset()
done = False
ml_path = []

while not done:
    action = env.agent.act(state)  # Use the trained policy to act
    next_state, _, done = env.step(action)
    ml_path.append(next_state)
    state = next_state

# Convert ml_path to a numpy array for easy handling
ml_path = np.array(ml_path)

# Check if ml_path has valid data
if ml_path.size == 0 or len(ml_path) < 2:
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



# Plot cube function
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


# Plotting and Animation

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
speed = 1 # distance per frame

#the following lines draw three stationary cubes at6 the starting point 
#cube1 = plot_cube(ax, create_cube_vertices(cube1_position, cube_size), color='orange')
#cube2 = plot_cube(ax, create_cube_vertices(cube2_position, cube_size), color='blue')
#cube3 = plot_cube(ax, create_cube_vertices(cube3_position, cube_size), color='green')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')



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
if ml_path.size > 0 and len(ml_path) >= 2:
    ax.plot(ml_path[:, 0], ml_path[:, 1], ml_path[:, 2], color='black', label="ML Path")
    print(f"ml_path length: {len(ml_path)}")
    print(f"ml_path data:\n{ml_path}")
else:
    print("Warning: ML path is not available or has insufficient points.")




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
speeds = [2.0, 1.2, 1.3]  # Different speeds for each cube

cube_plots = [
    plot_cube(ax, create_cube_vertices(cube_start_positions[i], cube_size), color=color)
    for i, color in enumerate(['orange', 'blue', 'green'])
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
SAFE_DISTANCE = 10.0  # Safe distance threshold between cubes
OFFSET_DISTANCE = 5.0  # Distance to offset for collision avoidance

# Define a small threshold for goal distance
GOAL_THRESHOLD = 1.0  # Adjust as needed based on environment scale

# Update `update_cubes` to capture end time when each cube reaches the goal
#collision_radius = np.sqrt(3 * (cube_size / 2) ** 2)
collision_radius = cube_size * np.sqrt(3) / 2  # Radius for cube-to-cube collision detection


# Define `offset_applied` globally if it should persist across frames
offset_applied = [False] * len(cube_plots)  # Initialize offset tracking for each cube

# Enhanced function to check collision-free paths that includes the cube's collision radius
def collision_free_expanded(X, start, end, radius):
    """
    Checks if the path from start to end is free of obstacles, considering the specified radius around the path.
    """
    step_vector = np.array(end) - np.array(start)
    steps = int(np.linalg.norm(step_vector) / (radius / 2)) + 1  # Increase granularity
    for step in range(steps + 1):
        position = np.array(start) + step * step_vector / steps
        if not X.collision_free(position - radius, position + radius):
            return False
    return True


# Flag and start time for return delay
waiting_at_goal = [False] * len(cube_plots)  # Track if each cube is waiting at the goal
return_start_time = [None] * len(cube_plots)  # Start time for the return wait



# Update the update_cubes function to ensure proper collision avoidance
# Update the `update_cubes` function to handle individual times
def update_cubes(num):
    global distances_covered, cube_times, waiting_at_goal, return_start_time

    for i, (cube, path, speed) in enumerate(zip(cube_plots, cube_positions, speeds)):
        # If the cube is waiting at the goal, check the delay and reverse direction if 5 seconds have passed
        if waiting_at_goal[i]:
            if time.time() - return_start_time[i] >= 5:  # Wait for 5 seconds
                waiting_at_goal[i] = False  # Reset waiting state
                distances_covered[i] = 0  # Reset distance for the reverse journey
                path.reverse()  # Reverse path to move back to origin
            else:
                continue  # Skip further processing for this cube until the wait is over

        # Update distance covered and compute new position
        distances_covered[i] += speed
        new_position = get_position_along_path(path, distances_covered[i])

        # Check if the cube has reached the end of the (forward or reversed) path
        distance_to_goal = np.linalg.norm(new_position[:3] - np.array(x_final_goal[:3]))
        cube_name = f'cube{i+1}'

        # Forward journey completion, prepare for wait at the goal
        if distance_to_goal < GOAL_THRESHOLD and cube_times[cube_name]['end'] is None and not waiting_at_goal[i]:
            cube_times[cube_name]['end'] = time.time()  # Record end time for the cube
            waiting_at_goal[i] = True  # Set the cube in a waiting state
            return_start_time[i] = time.time()  # Record time when waiting starts
            continue  # Skip further movement for this cube in this frame

        # Calculate elapsed time if end time is recorded, else current time - start
        elapsed_time = (
            cube_times[cube_name]['end'] - cube_times[cube_name]['start']
            if cube_times[cube_name]['end'] is not None
            else time.time() - cube_times[cube_name]['start']
        )

        # Update the on-screen text for each cube's time
        time_texts[i].set_text(f"Cube {i + 1} Time: {elapsed_time:.2f} s")

        # Collision avoidance logic: Adjusts position to avoid collisions
        for j, other_path in enumerate(cube_positions):
            if i != j:  # Skip self-comparison
                other_position = get_position_along_path(other_path, distances_covered[j])
                distance_to_other = np.linalg.norm(np.array(new_position) - np.array(other_position))
                
                # Avoid collision with a buffer distance equal to twice the cube size
                if distance_to_other < cube_size * 2:
                    # Ensure offset_direction is of type float64 to avoid casting issues
                    # Calculate the offset direction and normalize it if the norm is not zero
                    offset_direction = np.array(np.cross(np.array(new_position) - np.array(other_position), [0, 0, 1]), dtype=np.float64)
                    norm = np.linalg.norm(offset_direction)

                    # Avoid division by zero by only normalizing if the norm is greater than zero
                    if norm > 0:
                        offset_direction /= norm
                    else:
                        # Assign a fallback direction if the offset direction is zero
                        offset_direction = np.array([1, 0, 0], dtype=np.float64)  # Example fallback

                    new_position += offset_direction * (cube_size + 2)

        # Update vertices for each cube at `new_position`
        cube_vertices = create_cube_vertices(new_position, cube_size)

        # Update each line for the cube with new vertices
        for line, (p1, p2) in zip(cube, [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
        ]):
            line.set_data([cube_vertices[p1][0], cube_vertices[p2][0]],
                          [cube_vertices[p1][1], cube_vertices[p2][1]])
            line.set_3d_properties([cube_vertices[p1][2], cube_vertices[p2][2]])

    return [line for cube in cube_plots for line in cube] + time_texts




# Set up the animation with the updated frame count
longest_path_length = max(len(path) for path in paths)
ani = animation.FuncAnimation(fig, update_cubes, frames=longest_path_length, interval=100, blit=True)




# Add legend and show plot
ax.legend()
plt.show()


