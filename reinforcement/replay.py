from collections import deque
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque()
    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append(experience)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    def __len__(self):
        return len(self.buffer)
    def clear(self):
        self.buffer.clear()
    def __getitem__(self, idx):
        return self.buffer[idx]
    def __setitem__(self, idx, value):
        self.buffer[idx] = value
    def __iter__(self):
        return iter(self.buffer)
    def __next__(self):
        return next(iter(self.buffer))
    def __contains__(self, item):
        return item in self.buffer
    def __repr__(self):
        return f"ReplayBuffer({self.capacity})"
    def __str__(self):
        return f"ReplayBuffer with {len(self.buffer)} experiences"
    