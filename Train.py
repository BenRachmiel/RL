from DQN import DQNAgent
from Environment import SimpleEnvironment
from Visualization import visualize


def train_agent(agent, env, episodes=10000, max_steps=1000):
    best_reward = float('-inf')

    for episode in range(episodes):
        total_episode_reward = 0

        state = env.reset()
        positions = [env.position.copy()]
        orientations = [env.orientation]
        targets = [env.target.copy()]

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action - 1)  # Map actions [0, 1, 2] to [-1, 0, 1] for turning
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_episode_reward += reward
            positions.append(env.position.copy())
            orientations.append(env.orientation)
            targets.append(env.target.copy())

        agent.decay_epsilon()

        if total_episode_reward > best_reward:
            best_reward = total_episode_reward
            if episode >= min(episodes / 2, 50):
                agent.save_model("best.pth")
                print(f"New best episode {episode}: Total Reward = {total_episode_reward:.2f}")
                visualize(positions, orientations, targets, episode)

        if episode % 500 == 499:
            visualize(positions, orientations, targets, episode)


# Parameters
state_dim = 5  # [x, y, orientation, target_x, target_y]
action_dim = 3  # 0: turn left, 1: go straight, 2: turn right
agent = DQNAgent(state_dim, action_dim)
agent.load_model("best.pth")

# Training parameters
episodes = 2500
env = SimpleEnvironment(speed=0.5, angular_velocity=0.1, area_size=(100, 100))
train_agent(agent, env, episodes=episodes, max_steps=500)

print("Training complete!")
