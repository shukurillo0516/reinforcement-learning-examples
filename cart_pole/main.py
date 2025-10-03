from tqdm import tqdm  # Progress bar
import gymnasium as gym
from agent import QAgent
from utils import get_discrete_cart_pole_observation

learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
n_episodes = 10_000  # Number of hands to practice
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration

# Create environment and agent
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
obs, info = env.reset()
done = False

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False
    disc_obs = get_discrete_cart_pole_observation(obs)

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(disc_obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)
        if not reward == 1:
            print(reward)
        next_disc_obs = get_discrete_cart_pole_observation(next_obs)

        # Learn from this experience
        agent.update(disc_obs, action, reward, terminated, next_disc_obs)

        # Move to next state
        done = terminated or truncated
        disc_obs = next_disc_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()


# Create testing environment
env = gym.make("CartPole-v1", render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

test_agent = QAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    q_values=agent.q_values,
)

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(disc_obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        disc_obs = next_disc_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()
