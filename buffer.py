from collections import deque
import random

class MemoryBuffer:
    def __init__(self, buffer_size, mini_batch_size):
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.mini_batch_size)

    def __len__(self):
        return len(self.buffer)
