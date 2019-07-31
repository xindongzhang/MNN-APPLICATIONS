import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)
from onnx import checker
import onnx

syms   = './mnist-symbol.json'
params = './mnist-0000.params'

input_shape = (1,1,28,28)

onnx_file = './mnist.onnx'

# Invoke export model API. It returns path of the converted onnx model
converted_model_path = onnx_mxnet.export_model(syms, params, [input_shape], np.float32, onnx_file)

# Load onnx model
model_proto = onnx.load_model(converted_model_path)

# Check if converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)