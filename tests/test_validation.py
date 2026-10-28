"""Walk-forward folds must be chronological, expanding and purged."""
import numpy as np

from glacier.validation import block_mcc, walk_forward_folds


def test_folds_never_train_on_or_after_the_test_block():
    for train, test in walk_forward_folds(250, 700, 50, purge=1):
        assert train.max() < test.min() - 1        # one-day purge
        assert train.min() == 0                    # expanding window


def test_folds_cover_the_range_without_overlap():
    tests = np.concatenate([te for _, te in walk_forward_folds(250, 700, 50)])
    assert tests.tolist() == list(range(250, 700))


def test_block_mcc_uses_full_blocks_only():
    y = np.tile([0, 1], 60)
    assert len(block_mcc(y, y.astype(float), block=50)) == 2
