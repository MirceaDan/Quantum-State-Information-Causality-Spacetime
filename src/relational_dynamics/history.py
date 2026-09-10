"""Metadata separating computational labels from relational physics."""

from __future__ import annotations

from dataclasses import dataclass

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"


@dataclass(frozen=True)
class RelationalEvent:
    computational_index: int
    clock_reading: float | int | None
    external_time: float | None = None
    emergent_order: int | None = None
    component_label: str = MATHEMATICAL_DEFINITION


def assert_no_hidden_external_time(event: RelationalEvent) -> None:
    if event.external_time is not None:
        raise AssertionError("external classical time must not be a physical event field")
