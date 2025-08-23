import os
from collections import Counter
from typing import BinaryIO, Sequence, Iterator
import regex as re


PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SAFE_CHUNK_CHARACTERS = [" "]


def chunk_file(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """TODO"""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def chunk_str(text: str, chunk_size: int = 4096) -> Iterator[str]:
    i = 0
    while i < len(text):
        j = i + chunk_size
        while j < len(text) and text[j] not in SAFE_CHUNK_CHARACTERS:
            j += 1
        yield text[i:j]
        i = j


def split_str(split_special_tokens: Sequence[str], text: str) -> Iterator[tuple[str, str]]:
    """TODO"""
    if len(split_special_tokens) > 0:
        special_token_pattern = (
            "(" + "|".join((re.escape(special_token) for special_token in split_special_tokens)) + ")"
        )
        splits = re.split(special_token_pattern, text)
        if len(splits) % 2 == 1:
            splits.append("")
        for split_and_delimiter in zip(*[iter(splits)] * 2):
            yield split_and_delimiter
    else:
        yield (text, "")


def pretokenize(text: str) -> Iterator[bytes]:
    """TODO"""
    matches = re.finditer(PRETOKENIZE_PATTERN, text)
    for match in matches:
        yield text[match.start() : match.end()]


def pretokenize_as_counts(split_special_tokens: Sequence[str], text: str) -> Counter[str]:
    """TODO"""
    counter = Counter()
    for split, _ in split_str(split_special_tokens=split_special_tokens, text=text):
        counter.update(Counter(pretokenize(text=split)))
    return counter


def pretokenize_bytes_as_counts(split_special_tokens: Sequence[str], encoding: str, text: bytes) -> Counter[str]:
    """TODO"""
    return pretokenize_as_counts(
        split_special_tokens=split_special_tokens,
        text=text.decode(encoding),
    )


def merge_word(pair_to_merge: tuple[bytes, bytes], word: Sequence[bytes]) -> ...:
    token_bytes = pair_to_merge[0] + pair_to_merge[1]
    new_word = []
    i = 0
    while i < len(word) - 1:
        if (word[i], word[i + 1]) == pair_to_merge:
            new_word.append(token_bytes)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    if i == len(word) - 1:
        new_word.append(word[i])
    return new_word
