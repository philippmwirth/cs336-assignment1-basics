from pathlib import Path
from collections.abc import Sequence, Collection
import regex as re
from collections import Counter, defaultdict
import itertools
import multiprocessing
import dataclasses
import functools
import logging
import timeit

from cs336_basics.tokenizer import utils
from cs336_basics.tokenizer import priority_queue

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


@dataclasses.dataclass(slots=True)
class Word:
    word: Sequence[bytes]
    count: int

    def merge(self, pair_to_merge: tuple[bytes, bytes]) -> tuple[Counter[bytes], bytes]:
        token_bytes = pair_to_merge[0] + pair_to_merge[1]
        new_word = []
        updates = defaultdict(int)
        i = 0
        while i < len(self.word) - 1:
            if (self.word[i], self.word[i + 1]) == pair_to_merge:
                # We found our pair! Merge it
                new_word.append(token_bytes)
                if i > 0:
                    updates[(self.word[i - 1], token_bytes)] += 1
                    updates[(self.word[i - 1], self.word[i])] -= 1
                if i < len(self.word) - 2:
                    updates[(self.word[i + 1], self.word[i + 2])] -= 1
                    updates[(token_bytes, self.word[i + 2])] += 1
                # updates[(self.word[i], self.word[i+1])]
                i += 2
            else:
                new_word.append(self.word[i])
                i += 1
        if i == len(self.word) - 1:
            new_word.append(self.word[i])

        self.word = new_word
        return updates


def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: Sequence[str],
    num_processes: int = 8,
    encoding: str = "utf-8",
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """Trains a BytePairEncoding on the input file

    Args:
        input_path:
            Path to a text file with BPE tokenizer training data.
        vocab_size:
            Maximum final vocabulary size.
        special_tokens:
            List of strings to add to the vocabulary.
        num_processes:
            Number of processes.
        encoding:
            Byte encoding to use.

    Returns:
        A vocabulary mapping ids to bytes and a list of merges by order of
        creation.

    Raises:
        TBD
    """
    t0 = timeit.default_timer()
    word_counts = _get_word_counts_from_file(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
        encoding=encoding,
    )
    t1 = timeit.default_timer()
    logger.info(f"t-pretokenize: {t1-t0:.2f}s")
    word_counts = [Word([bytes([c]) for c in word.encode(encoding)], count) for word, count in word_counts.items()]
    vocabulary = _get_initial_vocab(
        special_tokens=special_tokens,
        initial_size_without_special_tokens=256,
        encoding=encoding,
    )
    pair_counts, pair_to_words = _get_initial_counts(
        word_counts=word_counts,
        encoding=encoding,
    )
    q = priority_queue.PriorityQueue(list(pair_counts.items()))
    merges = []
    while len(vocabulary) < vocab_size and (pair_to_merge := q.pop()) is not None:
        vocabulary[len(vocabulary)] = pair_to_merge[0] + pair_to_merge[1]
        merges.append(pair_to_merge)
        updates = defaultdict(int)
        for i in pair_to_words.pop(pair_to_merge):
            word = word_counts[i]
            count_updates = word.merge(pair_to_merge)
            for pair, update in count_updates.items():
                updates[pair] += update * word.count
                pair_to_words[pair].add(i)

        for pair, update in updates.items():
            q.update(pair=pair, count_delta=update)

    t2 = timeit.default_timer()
    logger.info(f"t-tokenize: {t2-t1:.2f}s")
    return vocabulary, merges


def _get_word_counts_from_file(
    input_path: Path,
    special_tokens: Sequence[str],
    num_processes: int = 8,
    encoding: str = "utf-8",
) -> Counter[bytes]:
    """TODO"""
    with input_path.open("rb") as f:
        boundaries = utils.chunk_file(
            file=f,
            desired_num_chunks=num_processes * 10,
            split_special_token="<|endoftext|>".encode(encoding),
        )
        f.seek(0)
        chunks = list(f.read(end - start) for start, end in zip(boundaries[:-1], boundaries[1:]))
        logging.info(f"{len(chunks)} chunks loaded")
        with multiprocessing.Pool(processes=num_processes) as pool:
            word_counts = Counter()
            pretokenize_fn = functools.partial(
                utils.pretokenize_bytes_as_counts,
                special_tokens,
                encoding,
            )
            for counts in pool.map(pretokenize_fn, chunks):
                word_counts.update(counts)
    return word_counts


def _get_initial_vocab(
    special_tokens: Sequence[str],
    initial_size_without_special_tokens: int,
    encoding: str,
) -> dict[int, bytes]:
    """TODO"""
    vocabulary = dict((i, (i).to_bytes(1)) for i in range(initial_size_without_special_tokens))
    for special_token in special_tokens:
        vocabulary[len(vocabulary)] = special_token.encode(encoding)
    return vocabulary


def _get_initial_counts(
    word_counts: Counter[bytes],
    encoding: str,
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], Collection[int]]]:
    """TODO"""
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for i, word in enumerate(word_counts):
        token_pair_counts = Counter((left, right) for left, right in zip(word.word[:-1], word.word[1:]))
        pair_counts.update({pair: word.count * pair_count for pair, pair_count in token_pair_counts.items()})
        for pair in token_pair_counts:
            pair_to_words[pair].add(i)
    return pair_counts, pair_to_words
