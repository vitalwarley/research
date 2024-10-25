import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from losses.scl import HardContrastiveLossV6  # Assuming the class is in a file named hcl_v6.py


def generate_test_data(batch_size, embedding_dim):
    embeddings = torch.randn(batch_size, embedding_dim, device="cuda")
    num_pairs = batch_size // 2
    positive_pairs = torch.stack([torch.arange(0, num_pairs * 2, 2), torch.arange(1, num_pairs * 2, 2)], dim=1).to(
        "cuda"
    )
    return embeddings, positive_pairs


def run_performance_test(batch_sizes, embedding_dim, num_iterations):
    hard_contrastive_loss = HardContrastiveLossV6().to("cuda")

    for batch_size in batch_sizes:
        print(f"Running performance test for batch size {batch_size}...")
        total_time = 0
        for _ in tqdm(range(num_iterations)):
            embeddings, positive_pairs = generate_test_data(batch_size, embedding_dim)

            torch.cuda.synchronize()
            start = time.perf_counter()

            loss = hard_contrastive_loss(embeddings, positive_pairs, "train")

            torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += end - start

        avg_time = total_time / num_iterations
        print(f"Batch size: {batch_size}, Average time: {avg_time:.6f} seconds")


def profile_performance(batch_size, embedding_dim):
    hard_contrastive_loss = HardContrastiveLossV6().to("cuda")
    embeddings, positive_pairs = generate_test_data(batch_size, embedding_dim)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("HardContrastiveLossV6"):
            loss = hard_contrastive_loss(embeddings, positive_pairs, "train")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    batch_sizes = [8, 16, 32]
    embedding_dim = 512
    num_iterations = 100

    print("Running performance test...")
    run_performance_test(batch_sizes, embedding_dim, num_iterations)

    print("\nRunning profiler for batch size 20...")
    profile_performance(20, embedding_dim)
