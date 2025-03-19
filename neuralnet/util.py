import numpy as np
class Parameter:
    def __init__(self, value):
        self.data = value
        self.grad = np.zeros_like(value)
    def get(self):
        return self.data

