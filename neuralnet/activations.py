from abc import ABC, abstractmethod
import numpy as np
class Activation(ABC):
    @abstractmethod
    def __call__(self, x):
        pass
    @abstractmethod
    def derivative(self, x):
        pass

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        _x = self.__call__(x)
        return _x * (1 - _x)

class ReLU(Activation):
    def __call__(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    def derivative(self, x):
        return 1 - np.tanh(x)**2

class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    def derivative(self, x, target=None, actions=None, rewards=None):
        softmax_out = self.__call__(x)  # Compute softmax
        
        if target is not None:
            # Cross-Entropy Gradient: (S - Y)
            return softmax_out - target
        
        elif actions is not None and rewards is not None:
            # Policy Gradient: (S - 1_a) * R
            grad = softmax_out.copy()
            grad[np.arange(len(actions)), actions] -= 1  # Adjust chosen actions
            grad *= rewards[:, np.newaxis]  # Scale by rewards
            return grad
        
        else:
            # Full Softmax Jacobian (not commonly used, but included for completeness)
            batch_size, num_classes = softmax_out.shape
            jacobians = np.zeros((batch_size, num_classes, num_classes))
            for i in range(batch_size):
                S = softmax_out[i].reshape(-1, 1)  # Column vector
                jacobians[i] = np.diagflat(S) - S @ S.T
            return jacobians  # Shape: (batch_size, num_classes, num_classes)
