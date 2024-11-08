import numpy as np
import time
import random
import sys
import subprocess
from rtree import index
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append('/home/dell/rrt-algorithms')

from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space2 import SearchSpace
from rrt_algorithms.utilities.plotting2 import Plot
from rrt_algorithms.dwa_algorithm.DWA3 import DWA
from rrt_algorithms.dqn_algorithm.DQN1 import DQNAgent
from rrt_algorithms.dqn_algorithm.DQN1 import DroneEnv
from rrt_algorithms.utilities.obstacle_generation2 import generate_random_cylindrical_obstacles 

# Define the search space dimensions
X_dimensions = np.array([(0, 100), (0, 100), (0, 100)])
X = SearchSpace(X_dimensions)

# Helper function to generate random positions
def generate_random_position(dimensions):
    x_min, x_max = dimensions[0]
    y_min, y_max = dimensions[1]
    z_min, z_max = dimensions[2]
    return (
        random.uniform(x_min, x_max),
        random.uniform(y_min, y_max),
        random.uniform(z_min, z_max)
    )
def path_length(path):
    return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))

# Function to try RRT to a goal and return the path
def try_rrt_to_goal(X, q, start, goal, max_samples, r, prc, obstacles):
    rrt = RRT(X, q, start, goal, max_samples, r, prc)
    path = rrt.rrt_search()
    if path is None:
        print(f"RRT failed to find a path to {goal}.")
    return path
# Initial and goal positions (Randomized)
x_init = generate_random_position(X_dimensions)
x_intermediate = generate_random_position(X_dimensions)
x_second_goal = generate_random_position(X_dimensions)
x_final_goal = generate_random_position(X_dimensions)

# Generate random obstacles
n = 0  # Number of random obstacles
Obstacles = generate_random_cylindrical_obstacles(X, x_init, x_final_goal, n)

# Define static obstacles
obstacles = [
    (10, 82, 0, 8, 15),  
    (80, 20, 0, 6, 19),   
    (10, 30, 0, 9, 15),   
    (80, 80, 0, 7, 22), 
    (50, 50, 0, 5, 22),
]

# Combine static and randomly generated obstacles into one list
all_obstacles = obstacles + Obstacles

# Initialize the SearchSpace with all obstacles
X = SearchSpace(X_dimensions, all_obstacles)

# RRT parameters
q = 600
r = 1
max_samples = 3024
prc = 0.1

# Function to create the distance matrix for a given path
def create_distance_matrix(path):
    num_points = len(path)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(np.array(path[i][:3]) - np.array(path[j][:3]))
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix

# Function to generate a .tsp file for LKH solver
def generate_tsp_file(path, filename="problem.tsp"):
    distance_matrix = create_distance_matrix(path)
    num_points = len(path)

    with open(filename, 'w') as f:
        f.write(f"NAME: LKH_TSP\nTYPE: TSP\nDIMENSION: {num_points}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        
        for i in range(num_points):
            f.write(" ".join(map(str, map(int, distance_matrix[i]))) + "\n")

        f.write("EOF\n")

# Function to run the LKH solver on the generated TSP file
def run_lkh_solver(tsp_file, param_file="lkh_params.par"):
    with open(param_file, 'w') as f:
        f.write(f"PROBLEM_FILE = {tsp_file}\nOPTIMUM = 0\nRUNS = 1\n")
    
    subprocess.run(["/home/dell/rrt-algorithms/examples/rrt/LKH-2.0.10/LKH", param_file])  # Assuming LKH is in the working directory and executable

# Function to parse the LKH solution
def parse_lkh_solution(solution_file="problem.sol"):
    tour = []
    with open(solution_file, 'r') as f:
        lines = f.readlines()
        start = lines.index("TOUR_SECTION\n") + 1
        end = lines.index("-1\n", start)
        tour = [int(line.strip()) - 1 for line in lines[start:end]]  # LKH uses 1-based indexing
    return tour

# Function to optimize the path using LKH
def lkh_optimize_path(path):
    tsp_file = "problem.tsp"
    generate_tsp_file(path, tsp_file)

    print("Running LKH solver...")
    run_lkh_solver(tsp_file)

    print("Parsing LKH solution...")
    solution_tour = parse_lkh_solution()

    optimized_path = [path[i] for i in solution_tour]
    return optimized_path

# Function to try RRT to a goal and return the path
def try_rrt_to_goal(X, q, start, goal, max_samples, r, prc, obstacles):
    rrt = RRT(X, q, start, goal, max_samples, r, prc)
    path = rrt.rrt_search()
    if path is None:
        print(f"RRT failed to find a path to {goal}.")
    return path

# Main RRT and LKH Optimization process
specified_time_intermediate = 20
specified_time_second = 30
specified_time_final = 50

print("RRT to first intermediate goal...")
path1 = try_rrt_to_goal(X, q, x_init, x_intermediate, max_samples, r, prc, all_obstacles)

if path1 is not None:
    print("RRT to second intermediate goal...")
    path2 = try_rrt_to_goal(X, q, x_intermediate, x_second_goal, max_samples, r, prc, all_obstacles)

    if path2 is not None:
        print("RRT to final goal...")
        path3 = try_rrt_to_goal(X, q, x_second_goal, x_final_goal, max_samples, r, prc, all_obstacles)

        if path3 is not None:
            path = path1 + path2[1:] + path3[1:]

            # Apply LKH optimization
            print("Applying LKH optimization to the path...")
            lkh_optimized_path = lkh_optimize_path(path)

            print("LKH optimization completed.")
        else:
            path = path1 + path2[1:]
    else:
        path = path1
else:
    path = None

if path is not None:
    print(f"Optimized Path: {lkh_optimized_path}")
else:
    print("RRT failed to find a valid path. Path length calculation skipped.")

if path is not None:
    dwa = DWA(dwa_params)
    optimized_path = []

    start_time = time.time()
    for i in range(len(path) - 1):
        start_point = path[i] + (0.0, 0.0)
        end_point = path[i + 1]

        local_path = dwa.plan(start_point, end_point, X, all_obstacles)
        optimized_path.extend(local_path)
    dwa_time = time.time() - start_time

    # Apply ALNS optimization to each segment of the path
    print("Applying ALNS optimization...")
    alns_optimized_path = []
    
    index_intermediate = find_closest_point(optimized_path, x_intermediate)
    index_second_goal = find_closest_point(optimized_path, x_second_goal)
    
    segment1 = optimized_path[:index_intermediate + 1]
    alns_optimized_segment1 = alns_optimize_path(segment1, X, all_obstacles, max_iterations=100)
    
    segment2 = optimized_path[index_intermediate:index_second_goal + 1]
    alns_optimized_segment2 = alns_optimize_path(segment2, X, all_obstacles, max_iterations=100)
    
    segment3 = optimized_path[index_second_goal:]
    alns_optimized_segment3 = alns_optimize_path(segment3, X, all_obstacles, max_iterations=100)
    
    alns_optimized_path = alns_optimized_segment1 + alns_optimized_segment2[1:] + alns_optimized_segment3[1:]

    def validate_and_correct_path(path):
        corrected_path = []
        for idx, point in enumerate(path):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                if isinstance(point[0], (list, tuple)):
                    point = [item for sublist in point for item in sublist]
                corrected_path.append(point[:3])
            else:
                raise ValueError(f"Point {idx} in path is incorrectly formatted: {point}")
        return corrected_path

    optimized_path = validate_and_correct_path(optimized_path)
    alns_optimized_path = validate_and_correct_path(alns_optimized_path)

# Continue the rest of your code as before...



# Metrics calculation
rrt_path_length = path_length(path)
dwa_path_length = path_length(optimized_path)
alns_path_length = path_length(alns_optimized_path)

rrt_smoothness = path_smoothness(path)
dwa_smoothness = path_smoothness(optimized_path)
alns_smoothness = path_smoothness(alns_optimized_path)

rrt_clearance = min_obstacle_clearance(path, all_obstacles)
dwa_clearance = min_obstacle_clearance(optimized_path, all_obstacles)
alns_clearance = min_obstacle_clearance(alns_optimized_path, all_obstacles)

average_velocity = dwa_params['max_speed']
rrt_energy = compute_energy_usage(path, average_velocity)
dwa_energy = compute_energy_usage(optimized_path, average_velocity)
alns_energy = compute_energy_usage(alns_optimized_path, average_velocity)


# Training the DQN
# Initialize the DroneEnv environment for path optimization
env = DroneEnv(X, x_init, x_final_goal, all_obstacles, dwa_params)
env.train(episodes=20)  # Train the DQN agent for 10 episodes

# Initialize the environment for testing
# Initialize the DroneEnv environment for path optimization
state = env.reset()
done = False
ml_path = []
agent = DQNAgent(env.state_size, env.action_size)

# Debugging: Track the shape and content of each state added to ml_path
while not done:
    action = agent.act(state)
    next_state, _, done = env.step(action)
    
    print(f"Current state shape: {state.shape}, next state shape: {next_state.shape}")
    
    if next_state.size > 0:
        ml_path.append(next_state)  # Store the next state if valid
    else:
        print("Warning: next_state is empty or invalid!")

    state = next_state

# After the loop, check the size of the ml_path
if len(ml_path) == 0:
    print("Error: ml_path is empty. No valid states were stored during the simulation.")
else:
    # Proceed with further processing
    print(f"ML path successfully collected {len(ml_path)} states.")

# Convert ml_path to a valid numpy array by flattening or ensuring consistent shape
ml_path_cleaned = []

# Ensure each element in ml_path has the correct shape before appending
for step in ml_path:
    step = np.asarray(step)  # Convert to array if it's not
    if step.ndim > 1:
        step = step.flatten()  # Flatten if it's multi-dimensional
    ml_path_cleaned.append(step)

# Convert cleaned path to a numpy array
ml_path_cleaned = np.array(ml_path_cleaned)

# Check if the resulting path has the expected shape
if ml_path_cleaned.ndim != 2 or ml_path_cleaned.shape[1] != 6:  # Assuming state size is 6
    raise ValueError(f"Expected ml_path_cleaned to have shape (N, 6), but got {ml_path_cleaned.shape}")

# Continue with plotting or further processing
print("ML path (cleaned):", ml_path_cleaned)

# Save the trained DQN model
agent.save("dqn_model_weights.h5")
print("DQN model weights saved as 'dqn_model_weights.h5'")


# Print Metrics
print(f"RRT Path Length: {rrt_path_length}")
print(f"DWA Optimized Path Length: {dwa_path_length}")
print(f"ALNS Optimized Path Length: {alns_path_length}")
print(f"RRT Path Smoothness: {rrt_smoothness}")
print(f"DWA Optimized Path Smoothness: {dwa_smoothness}")
print(f"ALNS Optimized Path Smoothness: {alns_smoothness}")
print(f"RRT Path Minimum Clearance: {rrt_clearance}")
print(f"DWA Optimized Path Minimum Clearance: {dwa_clearance}")
print(f"ALNS Optimized Path Minimum Clearance: {alns_clearance}")

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



# Time calculation function
def calculate_time_to_goal(path_length, speed):
    """Calculate the time it takes to reach the goal given the path length and speed."""
    if speed <= 0:
        raise ValueError("Speed must be positive to calculate time.")
    return path_length / speed

# Function to print time to reach the goal for each method
def print_time_to_goal(label, path_length, speed):
    time_to_goal = calculate_time_to_goal(path_length, speed)
    print(f"{label} Time to Goal: {time_to_goal:.2f} seconds")
    return time_to_goal

# Now include the time calculation and improvement comparison
if path is not None:
    rrt_path_length = path_length(path)
    dwa_path_length = path_length(optimized_path)
    alns_path_length = path_length(alns_optimized_path)

    # Assuming max speed for each method (adjusted by algorithms)
    rrt_speed = dwa_params['max_speed']  # RRT typically uses the DWA's max speed
    dwa_speed = dwa_params['max_speed']
    alns_speed = dwa_params['max_speed']
    
    # Machine learning (ML) path speed
    ml_speed = dwa_params['max_speed']  # Assuming ML also uses DWA's speed for comparison
    
    # Print and calculate the time for each algorithm
    rrt_time = print_time_to_goal("RRT", rrt_path_length, rrt_speed)
    dwa_time = print_time_to_goal("DWA Optimized", dwa_path_length, dwa_speed)
    alns_time = print_time_to_goal("ALNS Optimized", alns_path_length, alns_speed)
    
    # If ML path is valid, calculate its time to goal
    if ml_path.size > 0:
        ml_path_length = path_length(ml_path)
        ml_time = print_time_to_goal("Machine Learning Optimized", ml_path_length, ml_speed)
        ml_time = None
        print("ML path is not valid for time calculation.")
    
    # Improvement analysis
    print("\n--- Improvement Analysis ---")
    if ml_time is not None:
        improvement_from_rrt = rrt_time - ml_time
        improvement_from_dwa = dwa_time - ml_time
        improvement_from_alns = alns_time - ml_time
        
        print(f"Improvement from RRT: {improvement_from_rrt:.2f} seconds")
        print(f"Improvement from DWA: {improvement_from_dwa:.2f} seconds")
        print(f"Improvement from ALNS: {improvement_from_alns:.2f} seconds")
    else:
        print("ML time improvement calculation skipped due to invalid ML path.")

# The rest of the code (plotting, animation) remains the same as before.

# Plotting and Animation

# First figure: RRT, DWA, ALNS paths with ML path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(X_dimensions[0])
ax.set_ylim(X_dimensions[1])
ax.set_zlim(X_dimensions[2])

# Plot the RRT, DWA, and ALNS paths as dashed lines
ax.plot(*zip(*path), linestyle='--', color='red', label="RRT Path")
ax.plot(*zip(*optimized_path), linestyle='--', color='blue', label="DWA Optimized Path")
ax.plot(*zip(*alns_optimized_path), linestyle='--', color='green', label="ALNS Optimized Path")

# Plot the ML path
ax.plot(*ml_path.T, color='black', label="ML Path")

# Plot static obstacles
for obs in all_obstacles:
    x_center, y_center, z_min, z_max, radius = obs
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(z_min, z_max, 100)
    x = radius * np.outer(np.cos(u), np.ones(len(v))) + x_center
    y = radius * np.outer(np.sin(u), np.ones(len(v))) + y_center
    z = np.outer(np.ones(len(u)), v)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

# Plot start and goal points
ax.scatter(*x_init, color='magenta', label="Start")
ax.scatter(*x_intermediate, color='pink', label="Intermediate Goal")
ax.scatter(*x_second_goal, color='cyan', label="Second Goal")
ax.scatter(*x_final_goal, color='green', label="Final Goal")

# Animation of the ML path on the same figure
drone_path_line, = ax.plot([], [], [], color='orange', linewidth=2)

def update_line(num, ml_path, line):
    line.set_data(ml_path[:num, 0], ml_path[:num, 1])
    line.set_3d_properties(ml_path[:num, 2])
    return line,

ani = animation.FuncAnimation(
    fig, update_line, frames=len(ml_path), fargs=(ml_path, drone_path_line), interval=100, blit=True
)

# Add legend and show the figure
ax.legend()
plt.show()





