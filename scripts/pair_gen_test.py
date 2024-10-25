import time

import numpy as np
import torch


class PairGenerator:
    def generate_pairs_original(self, embeddings, positive_pairs):
        batch_size = embeddings.size(0)
        all_pairs = set((i, j) for i in range(batch_size) for j in range(batch_size) if i != j)
        positive_pairs_set = set((i.item(), j.item()) for i, j in positive_pairs)
        negative_pairs = list(all_pairs - positive_pairs_set)
        return torch.tensor(list(positive_pairs_set)), torch.tensor(negative_pairs)

    def generate_pairs_vectorized(self, embeddings, positive_pairs):
        batch_size = embeddings.size(0)

        # Create all pairs using NumPy
        all_pairs = np.array([(i, j) for i in range(batch_size) for j in range(batch_size) if i != j])

        # Convert positive_pairs to NumPy array
        positive_pairs_np = positive_pairs.numpy()

        # Create sets for faster operations
        all_pairs_set = set(map(tuple, all_pairs))
        positive_pairs_set = set(map(tuple, positive_pairs_np))

        # Get negative pairs
        negative_pairs = np.array(list(all_pairs_set - positive_pairs_set))

        return torch.from_numpy(positive_pairs_np), torch.from_numpy(negative_pairs)


def check_output_equality(output1, output2):
    positive1, negative1 = output1
    positive2, negative2 = output2

    print(f"Negative pairs 1: {negative1}")
    print(f"Negative pairs 2: {negative2}")

    # Check if positive pairs are equal
    positive_equal = torch.equal(positive1.sort()[0], positive2.sort()[0])

    # Check if negative pairs are equal (order doesn't matter)
    negative1_set = set(map(tuple, negative1.tolist()))
    negative2_set = set(map(tuple, negative2.tolist()))
    negative_equal = negative1_set == negative2_set

    return positive_equal and negative_equal


def test_performance(batch_size, num_positive_pairs):
    # Create dummy embeddings and positive pairs
    embeddings = torch.randn(batch_size, 128)  # Assuming 128-dimensional embeddings
    positive_pairs = torch.randint(0, batch_size, (num_positive_pairs, 2))

    pair_generator = PairGenerator()

    # Test original method
    start_time = time.time()
    original_output = pair_generator.generate_pairs_original(embeddings, positive_pairs)
    original_time = time.time() - start_time

    # Test vectorized method
    start_time = time.time()
    vectorized_output = pair_generator.generate_pairs_vectorized(embeddings, positive_pairs)
    vectorized_time = time.time() - start_time

    # Check output equality
    outputs_equal = check_output_equality(original_output, vectorized_output)

    print(f"Batch size: {batch_size}, Positive pairs: {num_positive_pairs}")
    print(f"Original method time: {original_time:.4f} seconds")
    print(f"Vectorized method time: {vectorized_time:.4f} seconds")
    print(f"Speedup: {original_time / vectorized_time:.2f}x")
    print(f"Outputs equal: {outputs_equal}")
    print()

    return outputs_equal


# Run performance tests
all_outputs_equal = all(
    [
        test_performance(4, 2),
        test_performance(5, 5),
    ]
)

print(f"All outputs equal across all tests: {all_outputs_equal}")
