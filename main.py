import gym
import time
import matplotlib.pyplot as plt
import numpy as np

from DQN.MADQN import MADQN
from PPO.MAPPO import MAPPO

MODEL = 'DQN'
ENV_NAME = 'ma_gym:Switch4-v0'
N_EPISODES = 3_000
LOG_PERIOD = 200
RENDER = True
TEST_EPISODES = 10

if __name__ == '__main__':

    # make environment
    env = gym.make(ENV_NAME)

    # choose model
    if MODEL == 'DQN':
        MA_model = MADQN(env)
    elif MODEL == 'PPO':
        MA_model = MAPPO(env)

    successful_run = False

    while not successful_run:
        # train agents
        train_rewards, successful_train_agents, successful_run = MA_model.train_agents(n_episodes=N_EPISODES, log_period=LOG_PERIOD, render=RENDER)

        if successful_run:
            # test agents
            test_scores, successful_test_agents = MA_model.test_agents(n_games=TEST_EPISODES, render=RENDER)

            plt.plot(np.arange(len(train_rewards))*50, train_rewards)
            plt.title(ENV_NAME[-10:-3] + ": Cumulative rewards")
            plt.xlabel('Epoch')
            plt.ylabel('Rewards')
            plt.show()

            plt.plot(np.arange(len(successful_train_agents))*50, successful_train_agents)
            plt.title(ENV_NAME[-10:-3] + ": Number of agents reaching their goal")
            plt.xlabel('Epoch')
            plt.ylabel('# succesful agents')
            plt.show()