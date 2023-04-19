import pandas as pd
import numpy as np
import gym

class MMASHEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, user_id):
        self.user_id = user_id
        self.user_info = pd.read_csv(f'Dataset/user_{user_id}/user_info.csv')
        self.user_stats = pd.read_csv(f'Dataset/user_{user_id}/user{user_id}.csv')
        self.sleep = pd.read_csv(f'Dataset/user_{user_id}/sleep.csv')
        self.questionnaire = pd.read_csv(f'Dataset/user_{user_id}/questionnaire.csv')

        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,))

        self.current_step = 0
        self.episode_reward = 0

    def reset(self):
        self.current_step = 0
        self.episode_reward = 0

        obs = self._get_obs()

        return obs

    def step(self, action):
        self.current_step += 1

        # Perform action and calculate reward
        reward = self._calculate_reward(action)

        # Update episode reward
        self.episode_reward += reward

        # Get next observation
        obs = self._get_obs()

        # Check if episode is done
        done = self.current_step == len(self.user_stats) - 1

        return obs, reward, done, {}

    def render(self, mode='human'):
        # Print current step and episode reward
        print(f'Step: {self.current_step}, Action: {self._get_obs()[0]}, Reward: {self.episode_reward}')

    def _get_obs(self):
        obs = [self.user_stats.loc[self.current_step, 'Activity'],
               self.user_stats.loc[self.current_step, 'Steps'],
               self.user_stats.loc[self.current_step, 'HR'],
               self.user_stats.loc[self.current_step, 'Inclinometer'],
               self.sleep.loc[0, 'Efficiency'] / 100]

        return np.array(obs)

    def _calculate_reward(self, action):
        """
        Calculates the reward based on the action taken.
        """
        # Get values from questionnaire and sleep data
        questionnaire_data = self.questionnaire.loc[0]
        sleep_data = self.sleep.loc[0]

        # Default reward value
        reward = 0

        # Switch statement to calculate reward based on the action
        if action == 1:  # sleeping
            reward += sleep_data['Efficiency'] / 100
        elif action == 2:  # laying down
            reward -= 0.1
        elif action == 3:  # sitting
            reward -= 0.05
        elif action == 4:  # light movement
            reward += 0.1
        elif action == 5:  # medium movement
            reward += 0.2
        elif action == 6:  # heavy movement
            reward += 0.3
        elif action == 7:  # eating
            reward -= 0.1
        elif action == 8:  # small screen usage
            reward -= 0.05
        elif action == 9:  # large screen usage
            reward -= 0.1
        elif action == 10:  # caffeinated drink consumption
            reward -= sleep_data['Latency']/10
        elif action == 11:  # smoking
            reward -= (((questionnaire_data['STAI1'] + questionnaire_data['STAI2'])/2) + questionnaire_data['Daily_stress'])/100
        elif action == 12:  # alcohol assumption
            reward -= (((questionnaire_data['STAI1'] + questionnaire_data['STAI2'])/2) + questionnaire_data['Daily_stress'])/100
        return reward


