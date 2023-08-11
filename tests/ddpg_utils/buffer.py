# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np


class RBuffer():
    def __init__(self, maxsize, statedim, naction):
        self.cnt = 0
        self.maxsize = maxsize
        self.state_memory = np.zeros((self.maxsize, *statedim))
        self.next_state_memory = np.zeros((self.maxsize, *statedim))
        self.action_memory = np.zeros((self.maxsize, naction))
        self.reward_memory = np.zeros(self.maxsize)
        self.done_memory = np.zeros(self.maxsize, dtype=bool)

    def storexp(self, state, action, reward, done, next_state):
        index = self.cnt % self.maxsize

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.cnt += 1

    def sample(self, batch_size):
        currentmaxSize = min(self.cnt, self.maxsize)

        batch = np.random.choice(currentmaxSize, batch_size, replace=False)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]
        return states, actions, rewards, dones, next_states
