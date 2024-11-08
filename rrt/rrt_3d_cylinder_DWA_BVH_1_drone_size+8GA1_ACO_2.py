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

from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space2 import SearchSpace
from rrt_algorithms.utilities.plotting2 import Plot
from rrt_algorithms.dwa_algorithm.DWA3 import DWA, FuzzyController
from rrt_algorithms.dqn_algorithm.DQN import DQNAgent
from rrt_algorithms.dqn_algorithm.DQN import DroneEnv
from rrt_algorithms.ACO_algorithm.ACO_2 import AntColony


from rrt_algorithms.utilities.obstacle_generation2 import generate_random_cylindrical_obstacles 

# Define the search space dimensions
X_dimensions = np.array([(0, 100), (0, 100), (0, 100)])
X = SearchSpace(X_dimensions)

# Initial and goal positions
x_init = (50, 50, -2)
x_intermediate = (50, 50, 100)
x_second_goal = (20, 60, 100)
x_final_goal = (0, 0, 0)

# Generate random obstacles
n = 7  # Number of random obstacles
Obstacles = generate_random_cylindrical_obstacles(X, x_init, x_final_goal, n)

# Define static obstacles
obstacles = [
    (10, 82, 0, 85, 15),  
    (80, 20, 0, 60, 19),   
    (10, 30, 0, 90, 15),   
    (80, 80, 0, 70, 22), 
    (50, 50, 0, 50, 22),
]

# Combine static and randomly generated obstacles into one list
all_obstacles = obstacles + Obstacles
# Initialize the SearchSpace with all obstacles
X = SearchSpace(X_dimensions, all_obstacles)

# RRT parameters
q = 30
r = 1
max_samples = 1024
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
# Initialize GA metrics
ga_path_length = None
ga_smoothness = None
ga_clearance = None

# Initialize ACO metrics
aco_path_length = None
aco_smoothness = None
aco_clearance = None

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
        
        # Calculate norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            print(f"Warning: Zero-length vector encountered at index {i}. Skipping curvature calculation for this segment.")
            continue  # Skip this iteration if a vector has zero length

        # Compute the cosine of the angle
        cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)  # Clamp value
        angle = np.arccos(cos_angle)
        total_curvature += angle

    return total_curvature

# Adaptive Large Neighborhood Search (ALNS) Functions
def dynamic_neighborhood_selection(neighborhoods, scores, iteration):
    # Adapt neighborhood exploration strategy based on progress
    if iteration % 10 == 0:
        # Every 10 iterations, randomize selection more to escape local minima
        return random.choice(neighborhoods)
    else:
        # Normal weighted choice based on scores
        return random.choices(neighborhoods, weights=scores.values())[0]



def alns_optimize_path(path, X, all_obstacles, max_iterations=100):
    neighborhoods = [segment_change, detour_addition, direct_connection]
    neighborhood_scores = {segment_change: 1.0, detour_addition: 1.0, direct_connection: 1.0}
    current_path = path.copy()

    for _ in range(max_iterations):
        # Select neighborhood adaptively
      #  neighborhood = random.choices(neighborhoods, weights=neighborhood_scores.values())[0]
        neighborhood = dynamic_neighborhood_selection(neighborhoods, neighborhood_scores, i)

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
    print("RRT to second intermediate goal...")
    rrt = RRT(X, q, x_intermediate, x_second_goal, max_samples, r, prc)
    path2 = rrt.rrt_search()

    if path2 is not None:
        print("RRT to final goal...")
        rrt = RRT(X, q, x_second_goal, x_final_goal, max_samples, r, prc)
        path3 = rrt.rrt_search()

        # Combine all paths
        if path3 is not None:
            path = path1 + path2[1:] + path3[1:]
        else:
            path = path1 + path2[1:]
    else:
        path = path1
else:
    path = None


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


# Metrics tracking
ga_path_length, aco_path_length = None, None
ga_smoothness, aco_smoothness = None, None
ga_clearance, aco_clearance = None, None

### Optimized GA Section



def generate_initial_population(size, path_length):
    return [np.random.rand(path_length, 3) * np.array([100, 100, 100]) for _ in range(size)]




def select_parents(population, fitness_scores):
    """ Select parents based on their fitness scores. """
    # Shift fitness scores if necessary to ensure all are non-negative
    min_fitness = min(fitness_scores)
    if min_fitness < 0:
        fitness_scores = [f - min_fitness for f in fitness_scores]  # Shift all scores to be positive

    total_fitness = sum(fitness_scores)
    
    # If all fitness scores are zero (can happen if no good solutions), set uniform probabilities
    if total_fitness == 0:
        probabilities = [1.0 / len(population)] * len(population)
    else:
        probabilities = [f / total_fitness for f in fitness_scores]

    parents_indices = np.random.choice(range(len(population)), size=2, p=probabilities)
    return population[parents_indices[0]], population[parents_indices[1]]

import numpy as np

def evaluate_fitness(path, obstacles):
    """ Calculate fitness based on path length and obstacle clearance. """
    path_len = path_length(path)
    clearance = min_obstacle_clearance(path, obstacles)
    
    if clearance < 0:
        fitness = 0  # Penalize paths inside obstacles
    else:
        fitness = 1.0 / (path_len + 1)  # Reward shorter paths
        fitness += clearance * 2.0  # Heavily reward higher clearance
    
    return max(fitness, 0)  # Ensure non-negative fitness

def mutate(path, mutation_rate):
    """ Mutate the path with a dynamic mutation rate and exploration direction. """
    for i in range(len(path)):
        if np.random.rand() < mutation_rate:
            random_shift = (np.random.rand(3) - 0.5) * 20  # Increased mutation range for more exploration
            path[i] += random_shift
            path[i] = np.clip(path[i], [0, 0, 0], [100, 100, 100])  # Ensure within bounds
    return path

def crossover(parent1, parent2):
    """ Improved crossover to promote diversity. """
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]], axis=0)
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]], axis=0)
    return child1, child2

def fitness_scaling(fitness_scores):
    """ Apply scaling to fitness scores to smooth selection. """
    scaled_fitness = np.array(fitness_scores) + 1.0  # Prevent division by zero
    return scaled_fitness

def genetic_algorithm(initial_path, obstacles, population_size=100, generations=200, elite_fraction=0.1, dynamic_mutation=True):
    """ Run GA with elitism, mutation diversity, and population diversity. """
    population = generate_initial_population(population_size, len(initial_path))
    best_path = None
    best_fitness = float('-inf')

    mutation_rate = 0.05  # Start with a higher mutation rate
    early_stop_threshold = 30
    no_improvement_generations = 0

    elite_size = int(population_size * elite_fraction)

    for generation in range(generations):
        fitness_scores = [evaluate_fitness(path, obstacles) for path in population]
        scaled_fitness_scores = fitness_scaling(fitness_scores)

        # Early stopping
        current_best_fitness = max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            no_improvement_generations = 0
        else:
            no_improvement_generations += 1

        if no_improvement_generations >= early_stop_threshold:
            print(f"Early stopping at generation {generation} due to no improvement.")
            break

        if dynamic_mutation:
            mutation_rate = max(0.01, mutation_rate * 0.95)

        # Apply elitism (carry over best solutions to next generation)
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

        new_population = sorted_population[:elite_size]  # Preserve top 'elite_size' paths

        # Generate the next population
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, scaled_fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        # Introduce random new paths for diversity
        for _ in range(population_size // 10):  # Introduce 10% random paths
            random_path = np.random.rand(len(initial_path), 3) * 100
            new_population.append(random_path)

        population = new_population

        # Track the best path across generations
        for path in population:
            fitness = evaluate_fitness(path, obstacles)
            if fitness > best_fitness:
                best_fitness = fitness
                best_path = path

    return best_path




# Apply DWA for local optimization along the RRT path
if path is not None:

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
    
    # Find the closest points to the goals
    index_intermediate = find_closest_point(optimized_path, x_intermediate)
    index_second_goal = find_closest_point(optimized_path, x_second_goal)
    
    # Segment 1: Start to Intermediate
    segment1 = optimized_path[:index_intermediate + 1]
    alns_optimized_segment1 = alns_optimize_path(segment1, X, all_obstacles, max_iterations=100)
    
    # Segment 2: Intermediate to Second Goal
    segment2 = optimized_path[index_intermediate:index_second_goal + 1]
    alns_optimized_segment2 = alns_optimize_path(segment2, X, all_obstacles, max_iterations=100)
    
    # Segment 3: Second Goal to Final Goal
    segment3 = optimized_path[index_second_goal:]
    alns_optimized_segment3 = alns_optimize_path(segment3, X, all_obstacles, max_iterations=100)
    
    # Combine the ALNS-optimized segments
    alns_optimized_path = alns_optimized_segment1 + alns_optimized_segment2[1:] + alns_optimized_segment3[1:]

# After the DWA and ALNS sections

# Use the last valid path from RRT as a starting point for ACO
# After the DWA and ALNS sections

# Use the last valid path from RRT as a starting point for ACO





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

# Final comparison metrics
print("\n--- Final Comparison ---")
print(f"RRT Path Length: {rrt_path_length}")
print(f"DWA Optimized Path Length: {dwa_path_length}")
print(f"ALNS Optimized Path Length: {alns_path_length}")
# Run GA
if path is not None:
    ga_optimized_path = genetic_algorithm(path, all_obstacles)

    # Check if GA returned a valid path
    if ga_optimized_path is not None and ga_optimized_path.size > 0:  # Ensure ga_optimized_path is a valid array and not empty
        ga_path_length = path_length(ga_optimized_path)
        ga_smoothness = path_smoothness(ga_optimized_path)
        ga_clearance = min_obstacle_clearance(ga_optimized_path, all_obstacles)

        # Calculate energy usage
        ga_energy = compute_energy_usage(ga_optimized_path, average_velocity)

        # Print GA metrics
        print(f"GA Optimized Path Length: {ga_path_length}")
        print(f"GA Optimized Path Smoothness: {ga_smoothness}")
        print(f"GA Optimized Path Minimum Clearance: {ga_clearance}")
        print(f"GA Energy Usage: {ga_energy}")
    else:
        print("GA failed to find a valid path.")

# Run ACO
aco = AntColony(X, all_obstacles, x_init, x_final_goal, num_ants=10, num_iterations=100, decay=0.1)
aco_optimized_path = aco.run()

# Check if ACO returned a valid path
if aco_optimized_path is not None and len(aco_optimized_path) > 0:  # Check if the path is valid
    # Ensure it's a 2D array or list of (x, y, z) coordinates
    if isinstance(aco_optimized_path, np.ndarray):
        aco_optimized_path = aco_optimized_path.reshape(-1, 3)  # Flatten if necessary
    elif isinstance(aco_optimized_path, list):
        aco_optimized_path = [point[:3] for point in aco_optimized_path if len(point) >= 3]

    # Plot the ACO path
    ax.plot(*zip(*aco_optimized_path), linestyle='--', color='purple', label="ACO Optimized Path")
else:
    print("ACO failed to find a valid path.")


    
    
    
    

# Check if GA path length is available
if ga_path_length is not None:
    print(f"GA Optimized Path Length: {ga_path_length}")
else:
    print("GA Optimized Path Length: Not available (failed to find a valid path)")

# Check if ACO path length is available
if aco_path_length is not None:
    print(f"ACO Optimized Path Length: {aco_path_length}")
else:
    print("ACO Optimized Path Length: Not available (failed to find a valid path)")

# Compare smoothness and clearance
if ga_smoothness is not None:
    print(f"GA Smoothness: {ga_smoothness}")
else:
    print("GA Smoothness: Not available")

if aco_smoothness is not None:
    print(f"ACO Smoothness: {aco_smoothness}")
else:
    print("ACO Smoothness: Not available")

print(f"RRT Clearance: {rrt_clearance}, GA Clearance: {ga_clearance}, ACO Clearance: {aco_clearance if aco_clearance is not None else 'Not available'}")
print(f"DWA Clearance: {dwa_clearance}, ALNS Clearance: {alns_clearance}")



# Final comparison metrics
print("\n--- Final Comparison ---")
print(f"RRT Path Length: {rrt_path_length}")
print(f"DWA Optimized Path Length: {dwa_path_length}")
print(f"ALNS Optimized Path Length: {alns_path_length}")
print(f"GA Optimized Path Length: {ga_path_length}")

# Compare smoothness and clearance
print(f"RRT Smoothness: {rrt_smoothness}, GA Smoothness: {ga_smoothness}")
print(f"DWA Smoothness: {dwa_smoothness}, ALNS Smoothness: {alns_smoothness}")

print(f"RRT Clearance: {rrt_clearance}, GA Clearance: {ga_clearance}")
print(f"DWA Clearance: {dwa_clearance}, ALNS Clearance: {alns_clearance}")
# Training the DQN
env = DroneEnv(X, x_init, x_final_goal, all_obstacles, dwa_params)
env.train(episodes=10)
# After training, use the learned policy in your DWA

# Initialize the DroneEnv environment for path optimization
state = env.reset()
done = False
ml_path = []
agent = DQNAgent(env.state_size, env.action_size)

while not done:
    action = agent.act(state)
    next_state, _, done = env.step(action)
    ml_path.append(next_state)
    state = next_state

# Convert ml_path to a numpy array
ml_path = np.array(ml_path)

# Check if ml_path has valid data
if ml_path.size == 0:
    print("ML path is empty. Ensure the DQN agent is working correctly.")
else:
    print("ML path:", ml_path)

# Save the trained DQN model
agent.model.save("dqn_model.keras")
print("DQN model saved as dqn_model.keras")

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

# Plotting and Animation

# First figure: RRT, DWA, ALNS paths with ML path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(X_dimensions[0])
ax.set_ylim(X_dimensions[1])
ax.set_zlim(X_dimensions[2])

# Plot the RRT, DWA, and ALNS paths as dashed lines

print("ACO Optimized Path:", aco_optimized_path)


ax.plot(*zip(*path), linestyle='--', color='red', label="RRT Path")
ax.plot(*zip(*optimized_path), linestyle='--', color='blue', label="DWA Optimized Path")
ax.plot(*zip(*alns_optimized_path), linestyle='--', color='green', label="ALNS Optimized Path")
# Plot the GA path
if ga_optimized_path is not None:
    ax.plot(*zip(*ga_optimized_path), linestyle='--', color='orange', label="GA Optimized Path")
# Plot the ACO path
if aco_optimized_path is not None:
    ax.plot(*zip(*aco_optimized_path), linestyle='--', color='purple', label="ACO Optimized Path")

# Legend
ax.legend()

# Legend
ax.legend()

# Plot the ML path
#ax.plot(*ml_path.T, color='black', label="ML Path")

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


