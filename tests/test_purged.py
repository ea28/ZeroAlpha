from dataclasses import dataclass
from datetime import datetime, timedelta, UTC

from zeroalpha.validation.purged import purge_overlapping_train, walk_forward_folds


@dataclass(frozen=True)
class Event:
    timestamp_utc: datetime
    t1: datetime


def test_purge_overlapping_train_events() -> None:
    events = [
        Event(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)),
        Event(datetime(2026, 1, 4, tzinfo=UTC), datetime(2026, 1, 5, tzinfo=UTC)),
        Event(datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 1, 11, tzinfo=UTC)),
    ]
    kept = purge_overlapping_train(
        events,
        [0, 1, 2],
        test_start=datetime(2026, 1, 2, tzinfo=UTC),
        test_end=datetime(2026, 1, 4, tzinfo=UTC),
        embargo=timedelta(days=1),
    )
    assert kept == [2]


def test_walk_forward_purges_train_and_calibration_overlap() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        Event(start + timedelta(hours=i), start + timedelta(hours=i + 4))
        for i in range(16)
    ]
    folds = walk_forward_folds(
        events,
        train_size=6,
        calibration_size=5,
        test_size=3,
        embargo=timedelta(0),
    )

    assert folds
    fold = folds[0]
    test_start = events[fold.test_indices[0]].timestamp_utc
    assert all(events[idx].t1 < test_start for idx in fold.calibration_indices)
    calibration_start = events[fold.calibration_indices[0]].timestamp_utc
    assert all(events[idx].t1 < calibration_start for idx in fold.train_indices)
