"""Purged walk-forward splitter for overlapping trade labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from zeroalpha.timeutils import ensure_utc


class TimedEvent(Protocol):
    timestamp_utc: datetime
    t1: datetime


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train_indices: list[int]
    calibration_indices: list[int]
    test_indices: list[int]


def purge_overlapping_train(
    events: list[TimedEvent],
    train_indices: list[int],
    test_start: datetime,
    test_end: datetime,
    embargo: timedelta,
) -> list[int]:
    test_start = ensure_utc(test_start)
    test_end = ensure_utc(test_end)
    embargo_end = test_end + embargo
    kept: list[int] = []
    for idx in train_indices:
        event = events[idx]
        start = ensure_utc(event.timestamp_utc)
        end = ensure_utc(event.t1)
        overlaps_test = start <= test_end and end >= test_start
        in_embargo = test_end < start <= embargo_end
        if not overlaps_test and not in_embargo:
            kept.append(idx)
    return kept


def purge_overlapping_indices(
    events: list[TimedEvent],
    indices: list[int],
    protected_start: datetime,
    protected_end: datetime,
    embargo: timedelta = timedelta(0),
) -> list[int]:
    return purge_overlapping_train(events, indices, protected_start, protected_end, embargo)


def walk_forward_folds(
    events: list[TimedEvent],
    *,
    train_size: int,
    calibration_size: int,
    test_size: int,
    embargo: timedelta,
) -> list[WalkForwardFold]:
    if min(train_size, calibration_size, test_size) <= 0:
        raise ValueError("fold sizes must be positive")
    ordered_indices = sorted(range(len(events)), key=lambda idx: events[idx].timestamp_utc)
    folds: list[WalkForwardFold] = []
    start = 0
    window = train_size + calibration_size + test_size
    while start + window <= len(ordered_indices):
        train = ordered_indices[start : start + train_size]
        cal = ordered_indices[start + train_size : start + train_size + calibration_size]
        test = ordered_indices[start + train_size + calibration_size : start + window]
        calibration_start = events[cal[0]].timestamp_utc
        calibration_test_end = max(max(events[idx].t1 for idx in cal), max(events[idx].t1 for idx in test))
        test_start = events[test[0]].timestamp_utc
        test_end = max(events[idx].t1 for idx in test)
        purged_train = purge_overlapping_train(
            events,
            train,
            calibration_start,
            calibration_test_end,
            embargo,
        )
        purged_cal = purge_overlapping_indices(events, cal, test_start, test_end, embargo)
        if purged_train and purged_cal:
            folds.append(WalkForwardFold(purged_train, purged_cal, test))
        start += test_size
    return folds
