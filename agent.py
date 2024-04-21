import random
from time import sleep

import numpy as np
import torch
from torch import nn

class ReplayMemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.MEMORY_SIZE = 1000
        self.BARCH_SIZE = 64

        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float32)
        self.all_a = np.random.randint(low=0, high=self.n_a, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_r = np.empty(shape=self.MEMORY_SIZE, dtype=np.float32)
        self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float32)

        self.t_max = 0
        self.t_memo = 0

    def add_memo(self, s, a, r, s_, done):
        self.all_s[self.t_memo] = s
        self.all_a[self.t_memo] = a
        self.all_r[self.t_memo] = r
        self.all_done[self.t_memo] = done
        self.all_s_[self.t_memo] = s_

        # self.t_max = self.t_max + 1 if self.t_max < self.MEMORY_SIZE else self.MEMORY_SIZE
        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo = (self.t_memo + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.t_max >= self.BARCH_SIZE:
            indexes = random.sample(range(self.t_max), self.BARCH_SIZE)
        else:
            indexes = range(self.t_max)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []

        for i in indexes:
            batch_s.append(self.all_s[i])
            batch_a.append(self.all_a[i])
            batch_r.append(self.all_r[i])
            batch_done.append(self.all_done[i])
            batch_s_.append(self.all_s_[i])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 88),
            nn.Tanh(),
            nn.Linear(88, n_output)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_tensor.unsqueeze(0))
        max_q_index = torch.argmax(input=q_values)
        action = max_q_index.detach().item()

        return action

class Agent:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.99
        self.learing_rate = 0.0001

        self.memo = ReplayMemory(n_input, n_output)

        self.online_net = DQN(self.n_input, self.n_output)
        self.target_net = DQN(self.n_input, self.n_output)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learing_rate)
