from collections.abc import Mapping, Sequence
from collections import defaultdict
from typing import Iterable, Iterator
import regex as re
import pickle
from pathlib import Path

# TODO: Move elsewhere
from cs336_basics.tokenizer import utils


class BPE:
    def __init__(
        self,
        vocab: Mapping[int, bytes],
        merges: Sequence[tuple[bytes, bytes]],
        special_tokens: Sequence[str] | None,
        encoding: str = "utf-8",
    ):
        self.vocab = vocab
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else []
        self.bytes_to_id = {val: key for key, val in self.vocab.items()}
        self.merges = merges
        self.encoding = encoding

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Sequence[str] | None = None):
        with Path(vocab_filepath).open("rb") as f:
            loaded_vocab = pickle.load(f)
        with Path(merges_filepath).open("rb") as f:
            loaded_merges = pickle.load(f)
        return cls(vocab=loaded_vocab, merges=loaded_merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        result = []
        for chunk in utils.chunk_str(text=text):
            splits_and_delimiters = utils.split_str(
                split_special_tokens=self.special_tokens,
                text=chunk,
            )
            for split, special_token in splits_and_delimiters:
                if len(split) == 0:
                    if special_token:
                        result.append(special_token.encode(self.encoding))
                    continue
                words = [[bytes([c]) for c in word.encode(self.encoding)] for word in utils.pretokenize(text=split)]
                for word in words:
                    for pair_to_merge in self.merges:
                        word = utils.merge_word(pair_to_merge=pair_to_merge, word=word)
                        if len(word) == 1:
                            break
                    result.extend(word)
                if special_token:
                    result.append(special_token.encode(self.encoding))

        return [self.bytes_to_id[token] for token in result]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text=text)

    def decode(self, ids: Sequence[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode(self.encoding, errors="replace")
