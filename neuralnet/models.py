class MLP:
    def __init__(self, *layers, loss, output_heads=1):
        self.layers = list(layers)
        self.loss = loss
        self.params = [param for layer in layers for param in layer.get_params()]
        self.output_heads = output_heads  # Number of output layers at the end

    def forward(self, inputs):
        for layer in self.layers[:-self.output_heads]:  # Process shared layers
            inputs = layer.forward(inputs)

        # Apply different output heads
        outputs = [layer.forward(inputs) for layer in self.layers[-self.output_heads:]]
        return outputs if len(outputs) > 1 else outputs[0]  # Return single output if only 1 head

    def backward(self, grads):
        # If multiple output layers, expect multiple gradients
        if not isinstance(grads, list):
            grads = [grads]  # Convert to list for consistency

        # Backpropagate through output layers
        shared_grad = None
        for grad, layer in zip(grads, reversed(self.layers[-self.output_heads:])):
            shared_grad = layer.backward(grad)

        # Backpropagate through shared layers
        for layer in reversed(self.layers[:-self.output_heads]):
            shared_grad = layer.backward(shared_grad)




        