from agent import PPOAgent
from environment import MMASHEnv
import utils

# Set the user ID to test the agent on
user_id = 12

# Initialize the environment and the agent
env = MMASHEnv(user_id)
agent = PPOAgent(env.action_space.n)

# Load the saved models
agent.load_models()

# Run test episodes
num_episodes = 100
episode_rewards = []
episode_lengths = []

for i in range(num_episodes):
    observation = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:
        action, _, _ = agent.choose_action(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        env.render()

    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

# Plot the rewards and episode lengths
utils.plot_results(episode_rewards, episode_lengths)
