from argparse import ArgumentParser

import numpy as np
import mxnet as mx
import onnx
import torch
from onnx import checker
import onnx_graphsurgeon as gs

from model import Model, PretrainModel
from tasks import init_fiw, init_parser
from pytorch_lightning import Trainer, seed_everything


def convert_mxnet():

    sym = "fitw2020/models/arcface_r100_v1-symbol.json"
    params = "fitw2020/models/arcface_r100_v1-0000.params"

    in_shapes = [(1, 3, 112, 112)]
    in_types = [np.float32]

    # Path of the output file
    onnx_file = "arcface_r100_v1.onnx"
    converted_model_path = mx.onnx.export_model(
        sym,
        params,
        in_shapes,
        in_types,
        onnx_file,
        verbose=True,
        run_shape_inference=True,
    )

    # Load the ONNX model and fix PReLU with graphsurgeon
    # Problem:
    #   - from inference with ort.InferenceSession
    #       onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Node:fc1 Output:fc1 [ShapeInferenceError] Mismatch between number of source and target dimensions. Source=4 Target=2
    #   - from onnxsim
    #       onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running PRelu node. Name:'relu0' Status Message: /onnxruntime_src/onnxruntime/core/providers/cpu/math/element_wise_ops.h:503 void onnxruntime::BroadcastIterator::Init(ptrdiff_t, ptrdiff_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1. 64 by 112
    # Problem solution: https://zhuanlan.zhihu.com/p/165294876
    model_proto = onnx.load_model(converted_model_path)
    graph = gs.import_onnx(model_proto)

    # https://github.com/NVIDIA/TensorRT/blob/master/tools/onnx-graphsurgeon/examples/04_modifying_a_model/modify.py
    # https://github.com/NVIDIA/TensorRT/blob/master/tools/onnx-graphsurgeon/examples/09_shape_operations_with_the_layer_api/generate.py
    prelu_nodes = [node for node in graph.nodes if node.op == "PRelu"]
    for node in prelu_nodes:
        constant_input = node.inputs[1]
        new_constant = gs.Constant(
            name=constant_input.name, values=constant_input.values.reshape(-1, 1, 1)
        )
        node.inputs = [node.inputs[0], new_constant]

    # onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Node:fc1 Output:fc1 [ShapeInferenceError] Mismatch between number of source and target dimensions. Source=4 Target=2
    # gemm_node = [node for node in graph.nodes if node.op == "Gemm"][0]
    # fc1_node = [node for node in graph.nodes if node.name == "fc1"][0]
    # fc1_node.inputs = [gemm_node.outputs[0], *fc1_node.inputs[1:]]
    # reshape_node = [node for node in graph.nodes if node.op == "Reshape"][0]
    # # refactor with: https://github.com/NVIDIA/TensorRT/blob/master/tools/onnx-graphsurgeon/examples/06_removing_nodes/remove.py
    # reshape_node_idx = graph.nodes.index(reshape_node)
    # del graph.nodes[reshape_node_idx]
    # graph.cleanup().toposort()

    # onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'' Status Message: GEMM: Dimension mismatch, W: {512,25088} K: 7 N:512
    # flatten_node = [node for node in graph.nodes if node.op == "Flatten"][0]
    # actual_flatten_output = flatten_node.outputs[0]
    # actual_flatten_output.shape = [1, 25088]
    # gemm_node.outputs[0].shape = [1, 512]
    onnx.save(gs.export_onnx(graph), converted_model_path)

    # TODO: fix flatten remaining error
    # Output:pre_fc1_data_flattened [ShapeInferenceError] Can't merge shape info. Both source and target dimension have values but they differ. Source=3584 Target=1 Dimension=0


def convert_pytorch():
    ### Export backbone.pth
    # Input to the model

    parser = init_parser()
    args = parser.parse_args(None)

    Trainer.from_argparse_args(
        args,
        devices=1,
        # accelerator="gpu",
    )

    args.model = "resnet101" if not args.model else args.model
    model = PretrainModel(**vars(args))
    model.eval()

    x = torch.randn(args.batch_size, 3, 112, 112, requires_grad=True)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "my_pretrained_model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=15,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["embeddings", "logits"],  # the model's output names
        verbose=True,
        dynamic_axes={
            "input": [0],
        },
    )

    # my_model_proto = onnx.load_model('my_pretrained_model.onnx')
    # inferred_model = onnx.shape_inference.infer_shapes(my_model_proto)
    # print(inferred_model.graph.value_info)


if __name__ == "__main__":
    # convert_mxnet()
    convert_pytorch()
