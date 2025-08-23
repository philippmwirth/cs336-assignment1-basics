import argparse
import pickle
from pathlib import Path
from cs336_basics.tokenizer import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Byte Pair Encoding (BPE) tokenizer.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a text file with BPE tokenizer training data.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path directory to store the outputs in.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10_000,
        help="Maximum final vocabulary size.",
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
    vocabulary, merges = train.train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
        encoding=args.encoding,
    )
    with (args.output_dir / "vocabulary.pkl").open("wb") as f:
        pickle.dump(vocabulary, f)
    with (args.output_dir / "merges.pkl").open("wb") as f:
        pickle.dump(merges, f)

    # Load merges and print them as a sanity check.
    with (args.output_dir / "vocabulary.pkl").open("rb") as f:
        loaded_vocab = pickle.load(f)
        max_index = max(loaded_vocab, key=lambda x: (len(loaded_vocab[x])))
        print(loaded_vocab[max_index])

    with (args.output_dir / "merges.pkl").open("rb") as f:
        loaded_merges = pickle.load(f)
