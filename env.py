import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from matplotlib.gridspec import GridSpec

class GridWorld:
    def __init__(self, height=3, width=4, block=[(1,1)], start=(2,0), goal=(0,3), trap=(1,3), trap_penalty=-1, is_slippery=False):
        self.height = height
        self.width = width
        self.block = block
        self.start = start
        self.actions = [0, 1, 2, 3]  # N, E, S, W
        self._actions = ['N', 'E', 'S', 'W']
        self.rewards = np.zeros((height, width))
        self.rewards[goal] = 1
        self.rewards[trap] = trap_penalty
        self.terminal_states = [goal, trap]
        self.is_slippery = is_slippery

    def is_terminal(self, state):
        return (state) in self.terminal_states

    def step(self, state, action):
        if self.is_terminal(state):
            return state
        transitions = self._get_transition_probs(action)
        next_states = []
        probs = []
        for a, p in transitions.items():
            ns = self._move(state, a)
            next_states.append(ns)
            probs.append(p)
        idx = random.choices(range(len(next_states)), weights=probs)[0]
        return next_states[idx], self.rewards[next_states[idx]]-0.04, self.is_terminal(next_states[idx])

    def _get_transition_probs(self, action):
        # 80% intended, 10% left, 10% right
        idx = action
        left = (idx - 1) % 4
        right = (idx + 1) % 4
        if self.is_slippery:
            return {action: 0.8, left: 0.1, right: 0.1}
        else:
            return {action: 1.0}

    def _move(self, state, action):
        i, j = state
        if action == 0: i -= 1
        elif action == 1: j += 1
        elif action == 2: i += 1
        elif action == 3: j -= 1
        if (i < 0 or i >= self.height or j < 0 or j >= self.width or (i,j) in self.block):
            return state
        return (i, j)
