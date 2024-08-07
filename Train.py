import os
import random
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from collections import deque
from matplotlib.animation import FuncAnimation, PillowWriter

from Environment import SimpleEnvironment


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = 64
        self.update_target_freq = 100

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learn_step_counter = 0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, path="best.pth"):
        """Save the current model to a file."""
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path="best.pth"):
        """Load model weights from a file."""
        if os.path.isfile(path):
            self.q_network.load_state_dict(torch.load(path))
            self.target_network.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}.")
        else:
            print(f"No model found at {path}. Starting with a new model.")


# Parameters
state_dim = 5  # [x, y, orientation, target_x, target_y]
action_dim = 3  # 0: turn left, 1: go straight, 2: turn right
agent = DQNAgent(state_dim, action_dim)

# Training parameters
episodes = 10000
env = SimpleEnvironment(speed=0.1, angular_velocity=0.1, area_size=(100, 100))

# Load existing model if available
agent.load_model("best.pth")

# Initialize the best reward to a very low number
best_reward = float('-inf')


def visualize_attempts_as_gif(positions, orientations, targets, episode, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    gif_filename = os.path.join(save_dir, f"episode_{episode}_{timestamp}.gif")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-1, 100)
    ax.set_ylim(-1, 100)
    ax.set_title(f'Episode {episode} - Agent Path')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    target_x, target_y = zip(*targets)
    target_plot, = ax.plot(target_x[0], target_y[0], 'ro', label='Target')

    agent_path, = ax.plot([], [], 'bo-', label='Agent Path')
    target_plot, = ax.plot([], [], 'ro', label='Target')
    direction_arrow = None

    def init():
        agent_path.set_data([], [])
        target_plot.set_data([], [])
        return agent_path, target_plot

    def update(frame):
        x_data, y_data = zip(*positions[:frame + 1])
        agent_path.set_data(x_data, y_data)

        # Update target position - repeat the position to mimic sequence format
        target_plot.set_data([targets[frame][0]], [targets[frame][1]])
        nonlocal direction_arrow
        if direction_arrow:
            direction_arrow.remove()

        pos = positions[frame]
        orient = orientations[frame]
        dx = 0.5 * np.cos(orient)
        dy = 0.5 * np.sin(orient)
        direction_arrow = ax.arrow(pos[0], pos[1], dx, dy, head_width=2, head_length=2, fc='blue', ec='blue')

        return agent_path, target_plot, direction_arrow

    ani = FuncAnimation(
        fig, update, frames=len(positions), init_func=init, blit=False, interval=100, repeat=False
    )
    ani.save(gif_filename, writer=PillowWriter(fps=30))
    print(f"Visualization saved as {gif_filename}.")


for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    positions = [env.position.copy()]
    orientations = [env.orientation]
    targets = [env.target.copy()]

    for _ in range(500):  # steps per episode
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action - 1)  # Map actions [0, 1, 2] to [-1, 0, 1] for turning
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward
        positions.append(env.position.copy())
        orientations.append(env.orientation)
        targets.append(env.target.copy())

        # Check for episode termination based on custom criteria (e.g., time limit)
        if done:
            break

    agent.decay_epsilon()

    if total_reward > best_reward:
        best_reward = total_reward
        if episode >= min(episodes / 2, 500):
            agent.save_model("best.pth")
            visualize_attempts_as_gif(positions, orientations, targets, episode)

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

print("Training complete!")
