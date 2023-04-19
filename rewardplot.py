from agent import PPOAgent
from environment import MMASHEnv
import matplotlib.pyplot as plt

# Set the user ID to test the agent on
user_id = 12

# Initialize the environment and the agent
env = MMASHEnv(user_id)
agent = PPOAgent(env.action_space.n)

# Load the saved models
agent.load_models()

# Run a test episode
observation = env.reset()
done = False
total_reward = 0
rewards = []
while not done:
    action, _, _ = agent.choose_action(observation)
    observation, reward, done, info = env.step(action)
    total_reward += reward
    rewards.append(total_reward)
    env.render()

# Plot the rewards obtained during the episode
plt.plot(rewards)
plt.xlabel('Time Step')
plt.ylabel('Cumulative Reward')
plt.title('Rewards obtained by the agent during the test episode')
plt.show()
