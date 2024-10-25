from argparse import ArgumentParser

import torch
import torch.cuda as cuda
import torch.nn as nn

from models.attention import FaCoRAttention
from models.facornet import FaCoR, FaCoRNetTask3


# Model Definitions
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = FaCoRNetTask3(model=FaCoR(FaCoRAttention()), list_dir="../datasets/rfiw2021-track3/txt")

    def forward(self, x):
        return self.model(x)


class ExtendedModel(nn.Module):
    def __init__(self, n_models):
        super(ExtendedModel, self).__init__()
        model = FaCoRNetTask3(model=FaCoR(FaCoRAttention()), list_dir="../datasets/rfiw2021-track3/txt")
        self.models = nn.ModuleList([model for _ in range(n_models)])

    def forward(self, x):
        ys = []
        for i, model in enumerate(self.models):
            y = model(x[i])
            ys.append(y)
        return ys


class ConcurrentModel(nn.Module):
    def __init__(self, n_models):
        super(ConcurrentModel, self).__init__()
        model = FaCoRNetTask3(model=FaCoR(FaCoRAttention()), list_dir="../datasets/rfiw2021-track3/txt")
        self.models = nn.ModuleList([model for _ in range(n_models)])

    def forward(self, x):
        streams = [torch.cuda.Stream() for _ in self.models]
        outputs = []

        for i, (model, stream) in enumerate(zip(self.models, streams)):
            with torch.cuda.stream(stream):
                outputs.append(model(x[i]))
        for stream in streams:
            stream.synchronize()
        return outputs


# Utility functions for timing and memory measurements
def measure_performance(model, args):
    inputs = [
        (torch.randn(args.batch_size, 3, 112, 112).cuda(), torch.randn(args.batch_size, 3, 112, 112).cuda())
        for _ in range(args.n_models)
    ]

    if args.model == "SimpleModel":
        inputs = inputs[0]

    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    _ = model(inputs)
    end_time.record()
    torch.cuda.synchronize()

    end_mem = torch.cuda.memory_allocated()
    time_ms = start_time.elapsed_time(end_time)
    memory_used = end_mem - start_mem
    return time_ms, memory_used


def print_model_memory_usage(model, name):
    torch.cuda.synchronize()  # Ensure all operations are completed
    initial_mem = torch.cuda.memory_allocated()  # Memory before loading model
    model.cuda()  # Move model to GPU to calculate its memory footprint
    torch.cuda.synchronize()  # Ensure model is loaded
    after_mem = torch.cuda.memory_allocated()  # Memory after loading model
    model_mem_usage = after_mem - initial_mem
    print(f"{name} memory usage: {model_mem_usage / 1024 ** 2:.2f} MB")
    return model_mem_usage


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--n_models", type=int, default=1)
    args.add_argument("--model", type=str, default="SimpleModel")
    args = args.parse_args()

    # Model initialization
    models = {
        "SimpleModel": SimpleModel(),
        "ExtendedModel": ExtendedModel(args.n_models),
        "ConcurrentModel": ConcurrentModel(args.n_models),
    }

    model = models[args.model]
    model.eval()
    model.cuda()

    if args.model == "SimpleModel":
        # Repeat the inference layer_count times to compare fairly
        times, mem_usages = zip(*[measure_performance(model, args) for _ in range(args.n_models)])
        avg_time = sum(times) / len(times)
        total_mem = sum(mem_usages) / len(mem_usages)
    else:
        avg_time, total_mem = measure_performance(model, args)

    print(
        f"{args.model}: Average Inference Time = {avg_time:.2f} ms, Total Memory Usage = {total_mem / 1024 ** 2:.2f} MB"
    )
    del model
    torch.cuda.empty_cache()
