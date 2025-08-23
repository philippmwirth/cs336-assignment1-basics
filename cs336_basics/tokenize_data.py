import argparse
import pickle
import numpy as np
from pathlib import Path
import multiprocessing
from cs336_basics.tokenizer import bpe
from cs336_basics.tokenizer import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text_path",
        type=Path,
        help="Path to the text file to tokenize.",
    )
    parser.add_argument(
        "vocabulary_path",
        type=Path,
        help="Path to the vocabulary pickle file.",
    )
    parser.add_argument(
        "merges_path",
        type=Path,
        help="Path to the merges pickle file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path directory to store the outputs in.",
    )
    parser.add_argument(
        "--special_tokens",
        nargs="*",  # 0 or more arguments
        default=["<|endoftext|>"],
        help="List of strings to add to the vocabulary. E.g., '<|endoftext|> <|pad|>'.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use for parallel processing (default: 8).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Byte encoding to use (default: 'utf-8').",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = bpe.BPE.from_files(
        vocab_filepath=args.vocabulary_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_tokens,
    )

    with args.text_path.open("rb") as f:
        boundaries = utils.chunk_file(
            f, desired_num_chunks=args.num_processes, split_special_token=args.special_tokens[0].encode(args.encoding)
        )
        chunks = list(f.read(end - start).decode(args.encoding) for start, end in zip(boundaries[:-1], boundaries[1:]))
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            tokens = []
            for tokens_chunk in pool.map(tokenizer.encode, chunks):
                tokens.extend(tokens_chunk)

    token_ids_uint16 = np.array(tokens).astype(np.uint16)
    np.save(args.output_dir / "tokens.npy", token_ids_uint16)
    print(f"Number of tokens: {len(token_ids_uint16)}")
