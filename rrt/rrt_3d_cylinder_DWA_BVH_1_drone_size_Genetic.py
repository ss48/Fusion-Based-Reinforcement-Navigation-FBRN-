import numpy as np
import time
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append('/home/dell/rrt-algorithms')

from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space2 import SearchSpace
from rrt_algorithms.utilities.plotting2 import Plot
from rrt_algorithms.dwa_algorithm.DWA3 import DWA
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

# Initial and goal positions (Randomized)
x_init = (50, 50, -2) #generate_random_position(X_dimensions)
x_intermediate = (50, 50, 100)
x_second_goal = (20, 60, 100)
x_final_goal = (0, 0, 0)
#x_intermediate = generate_random_position(X_dimensions)
#x_second_goal = generate_random_position(X_dimensions)
#x_final_goal = generate_random_position(X_dimensions)

# Generate random obstacles
n = 1  # Number of random obstacles
Obstacles = generate_random_cylindrical_obstacles(X, x_init, x_final_goal, n)

# Define static obstacles
obstacles = [
    (10, 82, 0, 80, 15),  
    (80, 20, 0, 60, 19),   
    (10, 30, 0, 19, 15),   
    (80, 80, 0, 70, 22), 
    (50, 50, 0, 15, 22),
]

# Combine static and randomly generated obstacles into one list
all_obstacles = obstacles + Obstacles

# Initialize the SearchSpace with all obstacles
X = SearchSpace(X_dimensions, all_obstacles)

# RRT parameters
q = 500
r = 1
max_samples = 3024
prc = 0.1

# DWA parameters
dwa_params = {
    'max_speed': 1.0,
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

# Utility functions for metrics

def path_smoothness(path):
    total_curvature = 0.0
    for i in range(1, len(path) - 1):
        vec1 = np.array(path[i]) - np.array(path[i-1])
        vec2 = np.array(path[i+1]) - np.array(path[i])
        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        total_curvature += angle
    return total_curvature

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
    i = random.randint(0, len(path) - 3)
    j = random.randint(i + 2, len(path) - 1)

    x_a = path[i][:3]
    x_b = path[j][:3]

    if X.collision_free(x_a, x_b, r):
        new_path = new_path[:i + 1] + new_path[j:]

    return new_path

def find_closest_point(path, goal):
    distances = [np.linalg.norm(np.array(p[:3]) - np.array(goal)) for p in path]
    return np.argmin(distances)

# RRT pathfinding to goals
specified_time_intermediate = 20
specified_time_second = 30
specified_time_final = 50

# Function to calculate path length
def path_length(path):
    return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))

# Function to try RRT to a goal and return the path
def try_rrt_to_goal(X, q, start, goal, max_samples, r, prc, obstacles):
    rrt = RRT(X, q, start, goal, max_samples, r, prc)
    path = rrt.rrt_search()
    if path is None:
        print(f"RRT failed to find a path to {goal}.")
    return path
    

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

# Apply ALNS optimization
def alns_optimize_path(path, X, all_obstacles, max_iterations=10):
    neighborhoods = [segment_change, detour_addition, direct_connection]
    neighborhood_scores = {segment_change: 1.0, detour_addition: 1.0, direct_connection: 1.0}
    current_path = path.copy()

    for _ in range(max_iterations):
        neighborhood = random.choices(neighborhoods, weights=neighborhood_scores.values())[0]
        new_path = neighborhood(current_path, X, all_obstacles)

        new_path_length = path_length(new_path)
        current_path_length = path_length(current_path)

        if new_path_length < current_path_length:
            current_path = new_path
            neighborhood_scores[neighborhood] *= 1.1  # Reward successful neighborhood
        else:
            neighborhood_scores[neighborhood] *= 0.9  # Penalize unsuccessful neighborhood

    return current_path

# Genetic Algorithm Functions
def generate_population(pop_size, path):
    population = [path.copy() for _ in range(pop_size)]
    for individual in population:
        random.shuffle(individual)
    return population

def fitness_function(path, X, all_obstacles):
    path_len = path_length(path)
    clearance = min_obstacle_clearance(path, all_obstacles)
    smoothness_penalty = path_smoothness(path)
    return path_len + (1.0 / clearance if clearance > 0 else float('inf')) + smoothness_penalty




def select_parents(population, fitnesses):
    tournament_size = 30
    selected_parents = []
    for _ in range(2):  # Select two parents
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected_parents.append(min(tournament, key=lambda x: x[1])[0])
    return selected_parents

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + [p for p in parent2 if p not in parent1[:point]]
    return child

def mutate(child, mutation_rate=0.1):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(child) - 1)
            child[i], child[j] = child[j], child[i]
    return child
def genetic_algorithm_optimize_path(X, all_obstacles, initial_path, max_generations=10, pop_size=20, mutation_rate=0.02):
    population = generate_population(pop_size, initial_path)
    fitnesses = [fitness_function(p, X, all_obstacles) for p in population]

    for generation in range(max_generations):
        new_population = []
        for _ in range(pop_size // 2):  # Generate pop_size/2 new children
            parent1, parent2 = select_parents(population, fitnesses)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population
        fitnesses = [fitness_function(p, X, all_obstacles) for p in population]

    best_solution = min(zip(population, fitnesses), key=lambda x: x[1])[0]
    return best_solution
# Try RRT to a goal and return the path

# Main execution of RRT, DWA, ALNS, and GA
path1 = try_rrt_to_goal(X, q, x_init, x_intermediate, max_samples, r, prc, all_obstacles)
if path1 is not None:
    path2 = try_rrt_to_goal(X, q, x_intermediate, x_second_goal, max_samples, r, prc, all_obstacles)
    if path2 is not None:
        path3 = try_rrt_to_goal(X, q, x_second_goal, x_final_goal, max_samples, r, prc, all_obstacles)
        path = path1 + path2[1:] + path3[1:]
    else:
        path = path1 + path2[1:]
else:
    path = path1

if path is not None:
    dwa = DWA(dwa_params)
    optimized_path = []
    for i in range(len(path) - 1):
        start_point = path[i] + (0.0, 0.0)
        end_point = path[i + 1]
        local_path = dwa.plan(start_point, end_point, X, all_obstacles)
        optimized_path.extend(local_path)

    alns_optimized_path = alns_optimize_path(optimized_path, X, all_obstacles, max_iterations=100)
    ga_optimized_path = genetic_algorithm_optimize_path(X, all_obstacles, optimized_path, max_generations=100, pop_size=50)

# Plotting the comparison
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 7))

ax.set_xlim(X_dimensions[0])
ax.set_ylim(X_dimensions[1])
ax.set_zlim(X_dimensions[2])

# Plot RRT, DWA, ALNS, and GA paths with distinct markers and styles
if path is not None:
    ax.plot(*zip(*path), linestyle='--', color='red', label="RRT Path", marker='o')
if optimized_path:
    ax.plot(*zip(*optimized_path), linestyle='--', color='blue', label="DWA Optimized Path", marker='^')
if alns_optimized_path:
    ax.plot(*zip(*alns_optimized_path), linestyle='--', color='green', label="ALNS Optimized Path", marker='s')
if ga_optimized_path:
    ax.plot(*zip(*ga_optimized_path), linestyle='--', color='purple', label="GA Optimized Path", marker='D')

# Plot start and goal points
ax.scatter(*x_init, color='magenta', label="Start")
ax.scatter(*x_intermediate, color='pink', label="Intermediate Goal")
ax.scatter(*x_second_goal, color='cyan', label="Second Goal")
ax.scatter(*x_final_goal, color='green', label="Final Goal")

# Plot obstacles
for obs in all_obstacles:
    x_center, y_center, z_min, z_max, radius = obs
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(z_min, z_max, 100)
    x = radius * np.outer(np.cos(u), np.ones(len(v))) + x_center
    y = radius * np.outer(np.sin(u), np.ones(len(v))) + y_center
    z = np.outer(np.ones(len(u)), v)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

# Add legends
ax.legend()
plt.title("Comparison of RRT, DWA, ALNS, and GA Optimized Paths")
plt.show()

# Genetic Algorithm Functions












# Main execution of RRT, DWA, ALNS, and GA
print("RRT to first intermediate goal...")
start_time = time.time()
path1 = try_rrt_to_goal(X, q, x_init, x_intermediate, max_samples, r, prc, all_obstacles)
rrt_time = time.time() - start_time
alns_optimized_path = []
if path1 is not None:
    print("RRT to second intermediate goal...")
    path2 = try_rrt_to_goal(X, q, x_intermediate, x_second_goal, max_samples, r, prc, all_obstacles)

    if path2 is not None:
        print("RRT to final goal...")
        path3 = try_rrt_to_goal(X, q, x_second_goal, x_final_goal, max_samples, r, prc, all_obstacles)

        # Combine all paths
        if path3 is not None:
            path = path1 + path2[1:] + path3[1:]
        else:
            path = path1 + path2[1:]
    else:
        path = path1
else:
    path = None

if path is not None:
    rrt_path_length = path_length(path)
    print(f"RRT Path Length: {rrt_path_length:.2f}")
else:
    print("Error: RRT failed to find a valid path. Path length calculation skipped.")

# Apply DWA for local optimization along the RRT path
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

    # Apply ALNS optimization to the path
    print("Applying ALNS optimization...")
    
    alns_optimized_path = alns_optimize_path(optimized_path, X, all_obstacles, max_iterations=100)

    # Apply GA optimization to the path
    print("Applying Genetic Algorithm optimization...")
    ga_optimized_path = genetic_algorithm_optimize_path(X, all_obstacles, optimized_path, max_generations=100, pop_size=50)

# Metrics calculation for comparison

if path is not None:
    rrt_path_length = path_length(path)
    dwa_path_length = path_length(optimized_path)
    alns_path_length = path_length(alns_optimized_path)
    ga_path_length = path_length(ga_optimized_path)

    print(f"RRT Path Length: {rrt_path_length}")
    print(f"DWA Optimized Path Length: {dwa_path_length}")
    print(f"ALNS Optimized Path Length: {alns_path_length}")
    print(f"Genetic Algorithm Optimized Path Length: {ga_path_length}")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Set plot limits
ax.set_xlim(X_dimensions[0])
ax.set_ylim(X_dimensions[1])
ax.set_zlim(X_dimensions[2])

# Plot RRT path
#if path is not None and len(path) > 0:
#    ax.plot(*zip(*path), linestyle='-', color='red', label="RRT Path", marker='o')

# Plot DWA optimized path
#if optimized_path is not None and len(optimized_path) > 0:
 #   ax.plot(*zip(*optimized_path), linestyle='-', color='blue', label="DWA Optimized Path", marker='o')

# Plot ALNS optimized path
#if alns_optimized_path is not None and len(alns_optimized_path) > 0:
 #   ax.plot(*zip(*alns_optimized_path), linestyle='-', color='green', label="ALNS Optimized Path", marker='o')

# Plot GA optimized path
#if ga_optimized_path is not None and len(ga_optimized_path) > 0:
#    ax.plot(*zip(*ga_optimized_path), linestyle='-', color='purple', label="GA Optimized Path", marker='o')

# Plot obstacles
for obs in all_obstacles:
    x_center, y_center, z_min, z_max, radius = obs
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(z_min, z_max, 100)
    x = radius * np.outer(np.cos(u), np.ones(len(v))) + x_center
    y = radius * np.outer(np.sin(u), np.ones(len(v))) + y_center
    z = np.outer(np.ones(len(u)), v)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

# Plot start, intermediate, second goal, and final goal points
ax.scatter(*x_init, color='magenta', label="Start", s=100)
ax.scatter(*x_intermediate, color='pink', label="Intermediate Goal", s=100)
ax.scatter(*x_second_goal, color='cyan', label="Second Goal", s=100)
ax.scatter(*x_final_goal, color='green', label="Final Goal", s=100)

# Add legend and show the figure
ax.legend(loc="upper right")
plt.title('Comparison of RRT, DWA, ALNS, and GA Optimized Paths')
plt.show()

