import torch
import torch.nn as nn
import torch.onnx
from train_mnist import MNIST

# A model class instance (class not shown)
model = MNIST()

# Load the weights from a file (.pth usually)
weights_path = './mnist.pth'
state_dict = torch.load(weights_path)

# Load the weights now into a model net architecture defined by our class
model.load_state_dict(state_dict)

# Create the right input shape (e.g. for an image)
input = torch.randn(1, 1, 28, 28)

torch.onnx.export(model, input, "mnist.onnx", verbose=True)


import onnx

# Load the ONNX model
model = onnx.load("mnist.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)