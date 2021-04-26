import time

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from DQN.DQNAgent import DQNAgent
from DQN.VDNet import VDNet
from DQN.parameters import agent_params, vdn_params

class MADQN:
    def __init__(self, env):

        # environment
        self.env = env
        self.obs_dim = env.observation_space[0].shape[0] + 1
        self.n_actions = env.action_space[0].n

        # training parameters
        self.batch_size = agent_params['batch_size']
        self.steps_per_episode = vdn_params['steps_per_episode']

        # initialize agents
        self.agents = [DQNAgent(self.obs_dim, self.n_actions, **agent_params) for _ in range(env.n_agents)]
        self.burnin_steps = agent_params['burnin_steps']

        # value decomposition network
        self.vd_net = VDNet(input_dim=self.batch_size, output_dim=self.n_actions)
        policy_net_params = []
        for agent in self.agents:
            policy_net_params += agent.policy_net.parameters()
        self.vd_optimizer = optim.Adam(params=policy_net_params, lr=vdn_params['lr'])


    def train_agents(self, n_episodes: int, steps_per_episode: int = 51, log_period: int=100, render: bool=True):
        """ Train """

        train_rewards = []
        exploited_rewards = []
        successful_agents = []
        minibatch_test_scores, minibatch_succesful_agents = [], []

        start_time = time.time()
        for ep in range(n_episodes):

            # initialize new episode
            ep_step = 0
            current_ep_rewards = 0
            obs = self.env.reset()
            dones = [False for _ in self.agents]


            while not all(dones):

                # select agent actions from observations
                actions = []
                with torch.no_grad():
                    for i, agent in enumerate(self.agents):
                        # augment observation with steps in environment
                        full_obs = np.array(obs[i] + [ep_step / self.steps_per_episode], dtype=np.float32)
                        actions.append(agent.get_action(full_obs, explore=True))

                # perform agent's actions and retrieve transition parameters
                next_obs, rewards, dones, info = self.env.step(actions)
                ep_step += 1

                # add agent rewards of transition
                current_ep_rewards += sum(rewards)

                rewards = [r if not d else 5.0 for r, d in zip(rewards, dones)]

                if render and not ep % log_period:
                    self.env.render()
                    time.sleep(0.05)

                #
                qvals_sum = torch.zeros((self.batch_size, 1))
                target_sum = torch.zeros((self.batch_size, 1))

                for a, agent in enumerate(self.agents):
                    # augment observations with steps in environment
                    agent_obs = np.array(obs[a] + [ep_step / self.steps_per_episode], dtype=np.float32)
                    agent_next_obs = np.array(next_obs[a] + [ep_step / self.steps_per_episode], dtype=np.float32)

                    # let agent take step and add
                    qvals, target = agent.step(agent_obs, actions[a], rewards[a], agent_next_obs, dones[a])

                    if not self.agents[0].start_training():
                        continue

                    qvals_sum += qvals
                    target_sum += target

                # update episode variables
                obs = next_obs

                # stop here during burn-in period
                if not self.agents[0].start_training():
                    continue

                if vdn_params['loss_func'] == 'Huber':
                    loss = F.smooth_l1_loss(qvals_sum, target_sum)
                elif vdn_params['loss_func'] == 'MSE':
                    loss = F.mse_loss(qvals_sum, target_sum)

                self.vd_optimizer.zero_grad()

                loss.backward()

                # clip gradients
                self.vd_net.clip_gradients(vdn_params['clip_val'])
                self.vd_optimizer.step()

            train_rewards.append(current_ep_rewards)

            # if ep == 350 and max(exploited_rewards) < 0:
            if ep == 350 and max(train_rewards) < 0:
                print("NO POSITIVE REWARDS AFTER 300 EPISODES")
                self.env.close()
                return exploited_rewards, successful_agents, 0

            # log test score and successfull agents very 25 episodes
            if not ep % 10:
                score, suc_agents = self.test_agents(1, False, False)
                minibatch_test_scores.append(score)
                minibatch_succesful_agents.append(suc_agents)

            if not ep % 50:
                exploited_rewards.append(np.mean(minibatch_test_scores))
                successful_agents.append(np.mean(minibatch_succesful_agents))
                minibatch_test_scores, minibatch_succesful_agents = [], []

            if not ep % log_period:
                print(f'\nEpisode {ep}, {time.time()-start_time:.0f} sec. ({(time.time()-start_time)/60:.1f} min.)')
                print(f'Mean last 5 test rewards: {exploited_rewards[-1]:.2f}')
                print(f'Mean last 5 number of agents successfull: {successful_agents[-1]:.2f}')

            if np.mean(successful_agents[-10:]) == self.env.n_agents:
                break

        self.env.close()

        print("\n--- Finished training ---")
        print(f"Total training time: {time.time()-start_time:.0f} sec. ({(time.time()-start_time)/60:.1f} min.)")
        return exploited_rewards, successful_agents, 1


    def test_agents(self, n_games: int, render: bool = True, done_training: bool = True):
        """

        :param env:
        :param n_games:
        :return:
        """

        game_rewards = []
        successful_agents = []

        for game in range(n_games):

            obs = self.env.reset()

            cum_rewards = 0

            dones = [False for _ in self.agents]

            ep_step = 0
            target_reached = 0

            while not all(dones):

                # select agent actions from observations
                actions = []
                for i, agent in enumerate(self.agents):
                    # augment observation with steps in environment
                    full_obs = np.array(obs[i] + [ep_step / self.steps_per_episode], dtype=np.float32)
                    actions.append(agent.get_action(full_obs, explore=False))

                obs, rewards, dones, info = self.env.step(actions)

                target_reached += rewards.count(5.0)

                if render:
                    self.env.render()

                cum_rewards += sum(rewards)
                ep_step += 1

            game_rewards.append(cum_rewards)
            successful_agents.append(target_reached)
            time.sleep(0.2)

        if done_training:
            print('\n=== Test performance ===')
            print(f'Mean: {np.mean(game_rewards):.1f} / '
                  f'Min: {np.min(game_rewards):.1f} / '
                  f'Max: {np.max(game_rewards):.1f}')

            self.env.close()

        return game_rewards, successful_agents
