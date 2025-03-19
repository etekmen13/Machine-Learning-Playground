from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def __call__(self, predictions, targets):
        pass
    @abstractmethod
    def gradient(self, predictions, targets):
        pass

class MSE(Loss):
    def __call__(self, predictions, targets):
        return np.mean((predictions - targets)**2)
    def gradient(self, predictions, targets):
        return 2 * (predictions - targets) / (len(predictions) if predictions is list else 1)

class CrossEntropy(Loss):
    def __call__(self, predictions, targets):
        return -np.sum(targets * np.log(predictions)) / (len(predictions) if predictions is list else 1)
    def gradient(self, predictions, targets):
        return -targets / predictions / (len(predictions) if predictions is list else 1)
    
