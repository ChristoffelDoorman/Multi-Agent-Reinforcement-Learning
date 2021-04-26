
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        """ Adds new transition to the replay buffer """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """ Sample batch from replay buffer """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
