import numpy as np
import torch

from DQN.DQNet import DQN
from DQN.ReplayBuffer import *


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, **hyperparams):

        # neural net dimensions
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # initialize hyperparameters
        self._init_hyperparams(hyperparams)
        self.epsilon = self.epsilon_start

        # initialize replay buffer
        self.memory = ReplayBuffer(capacity=self.replay_buffer_capacity)

        # policy and target approximation neural nets
        self.policy_net = DQN(obs_dim, act_dim, self.hidden_layer_dim)
        self.target_net = DQN(obs_dim, act_dim, self.hidden_layer_dim)

        self.t = 0

    def _decay_epsilon(self):
        """ Decrease exploration over time by exponentially decaying exploration threshold """
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon) * \
            np.exp(-(self.t - self.burnin_steps) / self.epsilon_decay_rate)

    def _store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """ Add transition to replay buffer as pytorch tensors """
        this_state = torch.tensor(state, dtype=torch.float)
        this_action = torch.tensor([action], dtype=torch.long)
        this_reward = torch.tensor([reward], dtype=torch.float)
        this_next_state = torch.tensor(next_state, dtype=torch.float)
        this_done = torch.tensor([done], dtype=torch.bool)
        self.memory.push(this_state, this_action, this_reward, this_next_state, this_done)

    def start_training(self):
        if self.t < self.burnin_steps or len(self.memory) < self.batch_size:
            return False
        else:
            return True

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Perform single agent's transition in the environment
        :param state: the observation of the agent
        :param action: the action of the agent
        :param reward: the reward earned by the agent
        :param next_state: the next state of the agent
        :param done: boolean denoting if agent has reached terminal state
        :return: loss
        """
        self.t += 1

        # store transition in resplay buffer
        self._store_transition(state, action, reward, next_state, done)

        # only start training after burn-in and when enough instances in replay buffer
        if self.start_training() == False:
            return 0, 0

        # decay exploration threshold
        self._decay_epsilon()

        # sample from replay buffer
        transitions = self.memory.sample(self.batch_size)
        return self.get_value_functions(transitions)

    def get_t(self):
        return self.t

    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action given current state.
        :param state: ((1,2)-array) state of the agent [x, y]
        :param explore: True when training, False when testing (exploiting)
        :return: action that maximizes q-value
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.act_dim - 1)
        else:
            # convert to pytorch tensor
            this_state = torch.tensor(state, dtype=torch.float)

            # no need for backprop during inference
            with torch.no_grad():
                # return action that
                this_q_vals = self.policy_net.forward(this_state)

                return this_q_vals.max(0).indices.item()



    def get_value_functions(self, sampled_transitions) -> float:
        """
        Train agent on sampled batch from replay buffer
        :param sampled_transitions: List[Transitions], containing sampled batch from replay buffer
        :return: loss
        """
        # merge samples from replay buffer into one batch
        batch = Transition(*zip(*sampled_transitions))
        batch_states = torch.stack(batch.state)
        batch_actions = torch.cat(batch.action)
        batch_rewards = torch.cat(batch.reward)
        batch_next_states = torch.stack(batch.next_state)
        batch_not_dones = ~torch.cat(batch.done) # transitions in replay buffer are already done

        # gather current q-values by action
        current_qvals = self.policy_net.forward(batch_states).gather(dim=1, index=batch_actions.unsqueeze(1))

        # compute target q-values
        target_qvals = self.target_net.forward(batch_next_states).max(1).values.detach()

        # compute target function = reward + discounted target Q(s',a')
        target = batch_rewards + self.gamma * target_qvals * batch_not_dones

        if not self.soft_update and not self.t % 500:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        elif self.soft_update:
            self.do_soft_update()

        return current_qvals, target.unsqueeze(1)


    def do_soft_update(self):
        """  """

        for param_target, param_local in zip(self.target_net.parameters(),
                                             self.policy_net.parameters()):

            param_target.data.copy_(self.tau*param_local + (1.0-self.tau)*param_target.data)

    def _init_hyperparams(self, hyperparams) -> None:
        """
        Initialize hyperparameters
        :param hyperparams: (dict) dictionary containing hyperparameters to overwrite
        :return:
        """
        # neural network
        self.hidden_layer_dim = 128

        # soft update parameter
        self.soft_update = False
        self.tau = 0.01

        self.replay_buffer_capacity = 5_000

        # exploration hyperparameters
        self.epsilon_start = 0.99
        self.epsilon_end = 0.05
        self.epsilon_decay_rate = 4_000

        self.gamma = 0.95
        self.clip_vals = 1.0

        self.batch_size = 64
        self.loss_func = 'Huber'
        self.lr = 0.005

        self.burnin_steps = 1_500

        # overwrite default parameters
        for param, val in hyperparams.items():
            if isinstance(val, str):
                exec('self.' + param + ' = "' + val + '"')
            else:
                exec('self.' + param + ' = ' + str(val))

        assert self.loss_func in ('MSE', 'Huber')
