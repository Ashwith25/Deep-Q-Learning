from dqn import DeepQNetwork
from env import GridWorld
from buffer import MemoryBuffer
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class DeepQLearningAgent():
    def __init__(
            self,
            learning_rate=0.01,
            gamma=0.95,
            epsilon=1,
            epsilon_decay=0.999,
            sync_rate=10,
            buffer_size=10000,
            batch_size=32,
            hidden_size=(64, 64),
            is_slippery=False
        ):
        self.env = GridWorld(is_slippery=is_slippery)
        self.state_size = self.env.height * self.env.width
        self.action_size = len(self.env.actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.sync_rate = sync_rate
        self.batch_size = batch_size
        self.buffer = MemoryBuffer(buffer_size, batch_size)
        self.policy_net = DeepQNetwork(self.state_size, hidden_size[0], hidden_size[1], self.action_size)
        self.target_net = DeepQNetwork(self.state_size, hidden_size[0], hidden_size[1], self.action_size)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        self.rewards_log = []
        self.epsilon_log = []

    def state_to_vector(self, state):
        i, j = state
        idx = i * self.env.width + j
        state_vec = torch.zeros(self.state_size)
        state_vec[idx] = 1.0
        return state_vec

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        state_vec = self.state_to_vector(state)
        with torch.no_grad():
            q_vals = self.policy_net(state_vec.unsqueeze(0))
        return q_vals.argmax(dim=1).item()

    def train(self, episodes):
        sync = 0
        print("Training started...")
        print("Initial Policy:")
        self.print_policy()
        for episode in tqdm(range(episodes)):
            step = 0
            # print("----------------------------")
            # print(f"Episode {episode+1}/{episodes}")
            # print(f"Epsilon: {self.epsilon:.4f}")
            # state = self.env.reset()
            state = self.env.start
            done = False
            total_reward = 0

            while not done and step < 25:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(state, action)
                self.buffer.add((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                sync += 1
                step += 1

            self.rewards_log.append(total_reward)
            # print(f"Episode {episode+1}: Total Reward: {total_reward:.2f}")

            if len(self.buffer) >= self.batch_size:
                mini_batch = self.buffer.sample()
                self.optimize(mini_batch, self.policy_net, self.target_net)

                self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
                self.epsilon_log.append(self.epsilon)

                if sync > self.sync_rate:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    sync = 0

        print("Training completed.")
        print("Final Policy:")
        self.print_policy()

    def optimize(self, mini_batch, policy_net, target_net):
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        state_vectors = []
        next_state_vectors = []
        for state, next_state in zip(states, next_states):
            state_vec = self.state_to_vector(state)
            state_vectors.append(state_vec)

            next_state_vec = self.state_to_vector(next_state)
            next_state_vectors.append(next_state_vec)

        state_batch = torch.stack(state_vectors)
        next_state_batch = torch.stack(next_state_vectors)
        action_batch = torch.tensor(actions)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32)

        current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = target_net(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = self.loss_fn(current_q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def print_policy(self):
        policy_grid = []
        for i in range(self.env.height):
            row = []
            for j in range(self.env.width):
                state = (i, j)
                if state in self.env.block:
                    row.append('B')
                elif state in self.env.terminal_states:
                    row.append('G' if state == self.env.terminal_states[0] else 'T')
                else:
                    state_vec = self.state_to_vector(state)
                    with torch.no_grad():
                        q_vals = self.policy_net(state_vec.unsqueeze(0))
                    best_action = q_vals.argmax(dim=1).item()
                    row.append(self.env._actions[best_action])
            policy_grid.append(row)

        for row in policy_grid:
            print(' '.join(row))

def hyperparameter_testing():
    """Run experiments varying batch size and plot rewards (raw + rolling mean).

    Produces a single plot with all 3 batch sizes overlaid. Each batch size shows:
    - Per-episode reward (low opacity) in a distinct color
    - Rolling mean (high opacity) in the same color to show the learning curve
    
    The figure is saved to `images/hyperparam_batch_sizes.png`.
    """
    # learning_rates = [0.001, 0.01, 0.1]
    # discount_factors = [0.75, 0.95, 0.99]
    # neural_sizes = [(32, 32), (64, 64), (128, 128)]
    batch_sizes = [16, 32, 64]
    episodes = 6000
    rolling_window = 50
    save_path = "images/hyperparam_batch_sizes_slippery.png"
    colors = ['blue', 'green', 'red']

    results = {}

    for batch_size in batch_sizes:
        print(f"Running experiment with batch_size={batch_size} (episodes={episodes})")
        # Create a fresh agent for this batch size
        agent = DeepQLearningAgent(hidden_size=(128, 128), learning_rate=0.001, gamma=0.75, is_slippery=True, batch_size=batch_size)

        # Train and collect rewards
        agent.train(episodes=episodes)
        rewards = np.array(agent.rewards_log)
        results[batch_size] = rewards

    # Create single plot with all learning rates
    fig, ax = plt.subplots(figsize=(14, 6))

    # print("Reward", results)
    # LEast reward for each learning rate
    for batch_size in batch_sizes:
        rewards = results[batch_size]
        print(f"Batch Size {batch_size}: Min Reward: {rewards.min():.2f}, Max Reward: {rewards.max():.2f}, Mean Reward: {rewards.mean():.2f}")

    for (batch_size, color) in zip(batch_sizes, colors):
        rewards = results[batch_size]
        
        # Plot raw rewards with low opacity
        ax.plot(
            np.arange(len(rewards)), 
            rewards, 
            alpha=0.15, 
            color=color, 
            linewidth=0.8
        )

        # Plot rolling mean with higher opacity
        if len(rewards) >= rolling_window:
            kernel = np.ones(rolling_window) / rolling_window
            rolling = np.convolve(rewards, kernel, mode="valid")
            x_rolling = np.arange(rolling_window - 1, rolling_window - 1 + len(rolling))
            ax.plot(
                x_rolling, 
                rolling, 
                linewidth=2.5, 
                color=color,
                label=f"Batch Size = {batch_size}"
            )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12, color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_title("Hyperparameter Test - Batch Size Variation (with Rolling Mean & Epsilon Decay)", fontsize=14)
    ax.grid(alpha=0.3)

    # Create secondary y-axis for epsilon
    ax2 = ax.twinx()
    
    # Plot epsilon from the first agent (epsilon is same for all since it's based on episodes)
    
    # Generate epsilon values to match episode count
    ax2.plot(
        np.arange(len(agent.epsilon_log)), 
        agent.epsilon_log, 
        color='purple', 
        linewidth=2, 
        linestyle='--',
        label='Epsilon (exploration rate)',
        alpha=0.7
    )
    ax2.set_ylabel("Epsilon (Exploration Rate)", fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim([0, 1.05])

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    fig.tight_layout()

    # Ensure images dir exists and save
    try:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved hyperparameter plot to: {save_path}")
    except Exception as e:
        print(f"Could not save figure: {e}")

    plt.show()

if __name__ == "__main__":
    # agent = DeepQLearningAgent(is_slippery=False)
    # agent.train(episodes=1000)

    hyperparameter_testing()