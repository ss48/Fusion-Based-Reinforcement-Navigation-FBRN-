import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import sys
import uuid
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
sys.path.append('/home/dell/rrt-algorithms')
import gc
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space2 import SearchSpace
from rrt_algorithms.utilities.plotting2 import Plot
from rrt_algorithms.dwa_algorithm.DWA3 import DWA 
from rrt_algorithms.dwa_algorithm.DWA3 import FuzzyController
from rrt_algorithms.utilities.obstacle_generation2 import generate_random_cylindrical_obstacles 
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
     
    def _build_model(self):
        model = Sequential()
        model.add(Dense(12, input_shape=(self.state_size,), activation='relu'))  # Reduce neurons
        model.add(Dense(12, activation='relu'))  # Reduce neurons
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy policy for action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action

        # Ensure state is of shape (1, state_size)
        state = state.reshape(1, self.state_size)  
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Best action

    def replay(self, batch_size):
        """Trains the network using randomly sampled experiences."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = next_state.reshape(1, self.state_size)  # Reshape next_state
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            
            state = state.reshape(1, self.state_size)  # Reshape state
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class DroneEnv:
    def __init__(self, X, start, goal, obstacles, dwa_params):
        self.X = X
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.dwa_params = dwa_params
        self.goal_x, self.goal_y, self.goal_z = goal  # Assuming goal is a tuple or list of 3 coordinates
        self.state_size = 6
        self.action_size = 6
        self.goal_threshold = 1.0  # Define the goal threshold here (1.0 unit, adjust as needed)
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.reset()

    def reset(self):
        # Assuming the drone's state consists of 6 variables (e.g., x, y, z, velocity, yaw, etc.)
        self.state = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        return self.state.reshape(1, -1)

    def step(self, action):
        # Use get_next_state to get the updated state after applying the action
        next_state = self.get_next_state(action)
    
        # Calculate reward based on next_state (this is just an example)
        reward = self.calculate_reward(next_state)
    
        # Check if the episode is done (e.g., reached goal or failed)
        done = self.is_done(next_state)

        return next_state, reward, done

    def is_done(self, next_state):
        """
        Check if the drone has reached the goal or failed.
        """
        current_position = next_state[:3]  # Extract the drone's (x, y, z) position from the state
        goal_position = np.array([self.goal_x, self.goal_y, self.goal_z])  # Goal position as a numpy array
        
        # Calculate the distance to the goal
        distance_to_goal = np.linalg.norm(goal_position - current_position)
        
        # If the drone is within the goal threshold, return True to indicate that the episode is done
        if distance_to_goal < self.goal_threshold:
            return True
        
        # Optionally, add more failure conditions (e.g., drone crashes into an obstacle or time limit is exceeded)
        
        # If the goal is not reached and no failure condition, return False
        return False

    def get_next_state(self, action):
    # Ensure action is an array or list
        action = np.array(action).flatten()  # Convert to a flat array if needed

        next_state = np.zeros(6)  # Assuming your state has 6 dimensions (x, y, z, velocity, yaw, etc.)

    # Make sure the action array has the right number of elements before modifying the state
        if len(action) >= 1:
            next_state[0] += action[0]  # Modify x-coordinate
        if len(action) >= 2:
            next_state[1] += action[1]  # Modify y-coordinate (if applicable)
        if len(action) >= 3:
            next_state[2] += action[2]  # Modify z-coordinate (if applicable)
    
    # Update other state variables based on action if necessary
    # For example, if action affects velocity or yaw, modify them as well
        if len(action) >= 4:
            next_state[3] += action[3]  # Modify velocity
        if len(action) >= 5:
            next_state[4] += action[4]  # Modify yaw (if applicable)
        if len(action) >= 6:
            next_state[5] += action[5]  # Modify any other state component (if applicable)

        return next_state

    def get_reward(self, next_state):
        # Define a reward function based on the next state
        distance_to_goal = np.linalg.norm(np.array(self.goal) - np.array(next_state[:3]))
        reward = -distance_to_goal  # Penalize the agent for being far from the goal
        return reward

    def is_goal_reached(self, next_state):
        # Check if the drone has reached the goal
        distance_to_goal = np.linalg.norm(np.array(self.goal) - np.array(next_state[:3]))
        return distance_to_goal < 1.0  # Define a threshold for goal proximity



    def map_action_to_control(self, action):
        """Maps the DQN action to a change in velocity or yaw rate."""
        velocity_change = [-0.1, 0.0, 0.1][action % 3]  # Simple action mapping
        yaw_change = 0  # Keep yaw_change static for simplicity
        return velocity_change, yaw_change

    
    def calculate_reward(self, next_state):
        goal_state = np.array([self.goal_x, self.goal_y, self.goal_z])  # Goal coordinates (shape: (3,))
        current_position = next_state[0][:3]  # Extract (x, y, z) from next_state (shape: (3,))
    
    # Now both goal_state and current_position should have shape (3,)
        distance_to_goal = np.linalg.norm(goal_state - current_position)
    
    # Define a simple reward function
        if distance_to_goal < 1.0:
            reward = 100  # Large reward for reaching the goal
        else:
            reward = -distance_to_goal  # Negative reward proportional to the distance from the goal
    
        return reward

        
    def perform_action(self, velocity_change, yaw_change):
        """Executes the action and returns the updated state, reward, and done flag."""
        state = self.state
        new_velocity = state[3] + velocity_change
        new_yaw_rate = state[4] + yaw_change

        state = np.array([state[0], state[1], state[2], new_velocity, new_yaw_rate, state[5]])

        dwa = DWA(self.dwa_params)
        local_path = dwa.plan(state, self.goal, self.X, self.obstacles)

        if not local_path:
            reward = -100  # Collision penalty
            done = True
            updated_state = state  # Maintain current state
        else:
            updated_state = np.array(local_path[-1])
            reward = self.compute_reward(updated_state, local_path)
            done = np.linalg.norm(np.array(updated_state[:3]) - np.array(self.goal[:3])) < 1.0  # Check if the goal is reached

        return updated_state.reshape(1, -1), reward, done

    def compute_reward(self, state, path):
        """Computes a reward based on the state and path."""
        distance_to_goal = np.linalg.norm(np.array(state[:3]) - np.array(self.goal[:3]))
        reward = -distance_to_goal  # Reward getting closer to the goal
        return reward

    def train(self, episodes, batch_size=32):
        """Trains the DQN agent using the environment."""
        for episode in range(episodes):
            state = self.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.agent.act(state)
                next_state, reward, done = self.step(action)
                total_reward += reward
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state

                self.agent.replay(batch_size)
                gc.collect()
            print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward}")


# Ensure all other parts of the code that handle the state are consistent with the 6-element structure.
        
    def min_obstacle_clearance(self, path, obstacles):
        min_clearance = float('inf')
        for point in path:
            for obs in obstacles:
                clearance = np.linalg.norm(np.array(point[:2]) - np.array(obs[:2])) - obs[4]
                if clearance < min_clearance:
                    min_clearance = clearance
        return min_clearance
    def path_smoothness(self, path):
        total_curvature = 0.0
        for i in range(1, len(path) - 1):
            vec1 = np.array(path[i][:3]) - np.array(path[i-1][:3])  # Use the first 3 elements if these are the coordinates
            vec2 = np.array(path[i+1][:3]) - np.array(path[i][:3])
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            total_curvature += angle
        return total_curvature
    
    def compute_energy_usage(self, path, velocity):
        energy = 0.0
        for i in range(1, len(path)):
            distance = np.linalg.norm(np.array(path[i][:3]) - np.array(path[i-1][:3]))
            energy += distance * velocity
        return energy
        



