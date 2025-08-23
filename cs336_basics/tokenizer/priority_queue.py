import dataclasses
import heapq
from collections.abc import Sequence
import heapq


@dataclasses.dataclass(slots=True)
class PairAndCount:
    pair: tuple[bytes, bytes]
    count: int
    valid: bool = True

    def __lt__(self, other):
        if not isinstance(other, PairAndCount):
            return NotImplemented

        if self.count != other.count:
            return self.count > other.count

        return self.pair > other.pair

    def invalidate(self):
        self.valid = False


class PriorityQueue:
    def __init__(self, pairs_and_counts: Sequence[tuple[tuple[bytes, bytes], int]]):
        self.q = []
        self.entries = dict()
        for pair, count in pairs_and_counts:
            self.add(pair=pair, count=count)

    def pop(self) -> tuple[bytes, bytes] | None:
        while len(self.q) > 0:
            item = heapq.heappop(self.q)
            if item.valid and item.count > 0:
                self.entries.pop(item.pair)
                return item.pair
        return None

    def add(self, pair: tuple[bytes, bytes], count: int):
        if pair in self.entries:
            old_item = self.entries.pop(pair)
            old_item.invalidate()
        if count <= 0:
            return
        item = PairAndCount(pair=pair, count=count)
        self.entries[pair] = item
        heapq.heappush(self.q, item)

    def update(self, pair: tuple[bytes, bytes], count_delta: int):
        if pair in self.entries and self.entries[pair].valid:
            old_count = self.entries[pair].count
        else:
            old_count = 0
        self.add(pair=pair, count=old_count + count_delta)
