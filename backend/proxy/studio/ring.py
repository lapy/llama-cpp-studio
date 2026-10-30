"""Small sequence-numbered rings and async fan-out used by the native proxy."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Deque, Generic, Iterable, TypeVar

T = TypeVar("T")

# Enqueued onto a subscriber queue to tell its iterator the ring detached it
# after the queue overflowed; the SSE response then closes for a replay.
_CLOSED = object()


@dataclass(frozen=True)
class RingItem(Generic[T]):
    sequence: int
    value: T


class EventRing(Generic[T]):
    """A bounded replay buffer with lossless fan-out for live subscribers.

    Subscriber queues are bounded: a slow consumer that falls ``overflow_limit``
    events behind is force-closed (its SSE handler ends, the client reconnects
    with Last-Event-ID and replays from the ring). Bounded queues keep one
    stalled reader from holding events for every other subscriber.
    """

    def __init__(self, capacity: int = 1000, overflow_limit: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._items: Deque[RingItem[T]] = deque(maxlen=capacity)
        self._next = 1
        self._subscribers: set[asyncio.Queue[RingItem[T] | object]] = set()
        self._overflow_limit = max(1, overflow_limit)

    def append(self, value: T) -> RingItem[T]:
        item = RingItem(self._next, value)
        self._next += 1
        self._items.append(item)
        for queue in tuple(self._subscribers):
            if queue.full():
                # Slow consumer: drop its oldest pending event to make room
                # for the termination sentinel. The iterator ends, the SSE
                # response closes, and the client resumes from its replay
                # cursor (Last-Event-ID) served from the ring itself.
                self._subscribers.discard(queue)
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - full() raced
                    pass
                queue.put_nowait(_CLOSED)  # type: ignore[arg-type]
                continue
            queue.put_nowait(item)
        return item

    def since(self, sequence: int = 0) -> list[RingItem[T]]:
        return [item for item in self._items if item.sequence > sequence]

    def values(self) -> list[T]:
        return [item.value for item in self._items]

    def clear(self) -> None:
        self._items.clear()

    async def subscribe(self, sequence: int = 0) -> AsyncIterator[RingItem[T]]:
        queue: asyncio.Queue[RingItem[T] | object] = asyncio.Queue(maxsize=self._overflow_limit)
        self._subscribers.add(queue)
        try:
            for item in self.since(sequence):
                if queue.full():
                    # The consumer is already behind before live delivery
                    # began; closing now lets it reconnect and step through
                    # the replay in smaller windows.
                    return
                yield item
            while True:
                received = await queue.get()
                if received is _CLOSED:
                    return
                yield received  # type: ignore[misc]
        finally:
            self._subscribers.discard(queue)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[RingItem[T]]:
        return iter(tuple(self._items))
