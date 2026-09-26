import pytest

from d3gs.split import holdout_split, nearest_train_index


def test_split_is_deterministic_and_disjoint():
    a = holdout_split(160, 8)
    b = holdout_split(160, 8)
    assert a == b
    train, test = a
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(160))


def test_split_known_values():
    train, test = holdout_split(160, 8)
    assert test == list(range(4, 160, 8))
    assert len(test) == 20 and len(train) == 140
    assert 0 in train and 159 in train  # clip ends stay in train


def test_split_rejects_degenerate_args():
    with pytest.raises(ValueError):
        holdout_split(10, 1)
    with pytest.raises(ValueError):
        holdout_split(10, 4, offset=4)


def test_nearest_train_index():
    train, _ = holdout_split(20, 8)
    assert nearest_train_index(4, train) == 3  # tie between 3 and 5 -> earlier
    assert nearest_train_index(12, [0, 8, 16]) == 8
