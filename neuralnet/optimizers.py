from abc import ABC, abstractmethod
import numpy as np
class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass

class Adam(Optimizer):
    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
    def __call__(self, model):
        self.params = model.params
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
    def zero_grad(self):
        for p in self.params:
            if hasattr(p, 'grad'):
                p.grad = np.zeros_like(p.grad)
            else:
                p.grad = np.zeros_like(p)
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * p.grad**2
            m_hat = self.m[i] / (1 - self.betas[0]**self.t)
            v_hat = self.v[i] / (1 - self.betas[1]**self.t)
            # print("m_hat", m_hat.shape)
            # print("v_hat", v_hat.shape)
            # print("p.data", p.data.shape)
            # print("p.grad", p.grad.shape)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)