import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MetricTracker:
    """Tracks and plots various metrics during training and evaluation."""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch = 0

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def plot(self):
        """Plot all tracked metrics."""
        plt.figure(figsize=(10, 6))
        for key, values in self.metrics.items():
            plt.plot(values, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Metrics over Epochs')
        plt.legend()
        plt.grid()
        plt.show()
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.epoch = 0
    