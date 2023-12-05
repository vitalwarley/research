from argparse import ArgumentParser


# Re-defining the file path and the necessary functions since there was a reset in the Python environment.
def parse_family_id(line):
    """Extract the family ID from a line."""
    parts = line.split("/")
    return parts[2]  # The family ID is the third part of the path.


def check_unique_families_per_n_rows_efficient(file_path, chunk_size=25):
    """Check if each set of chunk_size rows has only one occurrence of each family."""
    results = []
    with open(file_path, "r") as file:
        chunk = []
        for line in file:
            family_id = parse_family_id(line)
            chunk.append(family_id)

            # Every chunk_size lines, check for uniqueness and reset the chunk
            if len(chunk) == chunk_size:
                results.append(len(set(chunk)) == len(chunk))
                chunk = []

        # Check the last chunk if it's not empty and not a full set of chunk_size lines
        if chunk:
            results.append(len(set(chunk)) == len(chunk))

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file-path", type=str, default="input.txt")
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()

    # Execute the function and retrieve the results
    unique_family_check_results_efficient = check_unique_families_per_n_rows_efficient(args.file_path, args.chunk_size)
    if all(unique_family_check_results_efficient):
        print("All chunks have unique families.")
    else:
        print("Not all chunks have unique families.")
        print(
            "Number of chunks with unique families: "
            + f"{sum(unique_family_check_results_efficient)/len(unique_family_check_results_efficient)}"
        )
        non_unique_chunks = [i for i, x in enumerate(unique_family_check_results_efficient) if not x]
        print(f"First non-unique chunk: {non_unique_chunks[0]}")
