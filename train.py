from agent import PPOAgent
from environment import MMASHEnv

def train(i):
    env = MMASHEnv(user_id=x)
    n_actions = env.action_space.n

    agent = PPOAgent(n_actions=n_actions)

    print("Training for User ", x)

    for i in range(100):
        done = False
        obs = env.reset()

        while not done:
            action, log_prob, value = agent.choose_action(obs)
            new_obs, reward, done, _ = env.step(action)

            agent.store_transition(obs, action, log_prob, value, reward, done)

            obs = new_obs
            env.episode_reward += reward

        agent.learn()

        print(f'Episode {i} reward: {env.episode_reward}')
        agent.episode_reward = 0

        if i % 50 == 0:
            agent.save_models()

    env.close()

if __name__ == '__main__':
    for x in range(12,23):
        train(x)
