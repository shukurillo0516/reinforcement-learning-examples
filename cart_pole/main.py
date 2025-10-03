from tqdm import tqdm  # Progress bar
import gymnasium as gym
from agent import QAgent

learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
n_episodes = 10  # Number of hands to practice
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration

# Create environment and agent
env = gym.make("CartPole-v1", render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    # total_reward += reward
    # episode_over = terminated or truncated


# for episode in tqdm(range(n_episodes)):
#     # Start a new hand
#     obs, info = env.reset()
#     done = False

#     # Play one complete hand
#     while not done:
#         # Agent chooses action (initially random, gradually more intelligent)
#         action = agent.get_action(obs)

#         # Take action and observe result
#         next_obs, reward, terminated, truncated, info = env.step(action)

#         # Learn from this experience
#         agent.update(obs, action, reward, terminated, next_obs)

#         # Move to next state
#         done = terminated or truncated
#         obs = next_obs

#     # Reduce exploration rate (agent becomes less random over time)
#     agent.decay_epsilon()
