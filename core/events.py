"""Event queue primitives for online scheduling simulation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum


class EventType(IntEnum):
    JOB_ARRIVAL = 1
    OP_COMPLETION = 2
    CHECKPOINT = 3


@dataclass(order=True, frozen=True)
class Event:
    time: int
    event_type: EventType
    job_id: int = -1
    op_id: int | None = field(default=None, compare=False)
    machine_id: int | None = field(default=None, compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek(self) -> Event | None:
        return self._heap[0] if self._heap else None

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
