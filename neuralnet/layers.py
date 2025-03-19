from abc import ABC, abstractmethod
import numpy as np
from neuralnet.util import Parameter
class Layer:
    def __init__(self, in_features, out_features, activation=None):
        self.weights = Parameter(np.random.randn(in_features, out_features) * 0.01)  # Small random init
        self.bias = Parameter(np.zeros(out_features))
        self.activation = activation
        self.inputs = None
        self.z = None
        self.a = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights.get()) + self.bias.get()
        self.a = self.activation(self.z) if self.activation is not None else self.z
        return self.a  # Only return post-activation output

    def backward(self, grad):
        if self.inputs is None:
            raise ValueError("No input data. Call forward first.")
        # Apply activation function derivative if there is an activation
        if self.activation is not None and hasattr(self.activation, "derivative"):
            grad = grad * self.activation.derivative(self.z)

        # Compute gradients for weights and bias
        # print("inputs", self.inputs.shape)
        # print("grad", grad.shape)
        # print("weights", self.weights.get().shape)
        # print("bias", self.bias.get().shape)
        self.weights.grad += np.outer(self.inputs.T, grad)
        self.bias.grad += np.sum(grad, axis=0, keepdims=True)

        # Compute gradient w.r.t. inputs for previous layer
        grad_input = np.dot(grad, self.weights.get().T)
        return grad_input  # Return this so previous layers can backpropagate
    def get_params(self):
        return [self.weights, self.bias]
   