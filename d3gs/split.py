"""Deterministic train / held-out split over a time-ordered frame list."""
from __future__ import annotations


def holdout_split(n_frames: int, every: int, offset: int | None = None) -> tuple[list[int], list[int]]:
    """Hold out every ``every``-th frame for evaluation.

    Held-out indices are ``offset, offset+every, ...`` (default offset = every // 2
    so the first/last frames, which only have one-sided neighbours, stay in train).
    Pure function of its arguments: no randomness, same output every call.
    """
    if every < 2:
        raise ValueError("every must be >= 2 (otherwise nothing is left to train on)")
    if n_frames <= 0:
        return [], []
    off = every // 2 if offset is None else offset
    if not 0 <= off < every:
        raise ValueError("offset must be in [0, every)")
    test = list(range(off, n_frames, every))
    test_set = set(test)
    train = [i for i in range(n_frames) if i not in test_set]
    return train, test


def nearest_train_index(test_idx: int, train: list[int]) -> int:
    """Train frame closest in time (ties -> the earlier frame)."""
    return min(train, key=lambda i: (abs(i - test_idx), i))
