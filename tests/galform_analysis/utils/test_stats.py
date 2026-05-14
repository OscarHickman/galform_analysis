import numpy as np
import pytest

from galform_analysis.utils.stats import (
    positive_count,
    positive_percentile,
    positive_se,
    positive_std,
)

# Shape: (4 runs × 3 bins).  Values deliberately include zeros and negatives.
_DATA = np.array(
    [
        [1.0, -1.0, 2.0],
        [3.0,  0.0, 4.0],
        [5.0,  2.0, 6.0],
        [-2.0, 3.0, -1.0],
    ]
)


def test_positive_count_values():
    counts = positive_count(_DATA)
    assert counts[0] == 3   # 1, 3, 5
    assert counts[1] == 2   # 2, 3  (0 and -1 excluded)
    assert counts[2] == 3   # 2, 4, 6


def test_positive_count_all_nonpositive():
    data = np.array([[-1.0, 0.0], [-2.0, 0.0]])
    np.testing.assert_array_equal(positive_count(data), [0, 0])


def test_positive_std_matches_numpy():
    # col 1 positives: [2, 3] → std(ddof=0) = 0.5
    std = positive_std(_DATA)
    assert abs(std[1] - np.std([2.0, 3.0])) < 1e-10


def test_positive_std_nan_when_no_positives():
    data = np.array([[-1.0], [0.0]])
    assert np.isnan(positive_std(data)[0])


def test_positive_percentile_median():
    pct = positive_percentile(_DATA, q=50)
    # col 0 positives: [1, 3, 5] → median = 3
    assert abs(pct[0] - 3.0) < 1e-10


def test_positive_percentile_nan_col():
    data = np.array([[-1.0, 2.0], [-2.0, 3.0]])
    pct = positive_percentile(data, q=50)
    assert np.isnan(pct[0])
    assert not np.isnan(pct[1])


def test_positive_se_formula():
    # se = std / sqrt(n) over positive values
    std = positive_std(_DATA[:, :1])
    n = positive_count(_DATA[:, :1])
    expected = std / np.sqrt(n)
    np.testing.assert_allclose(positive_se(_DATA[:, :1]), expected)


def test_positive_se_nan_when_no_positives():
    data = np.array([[-1.0], [0.0]])
    assert np.isnan(positive_se(data)[0])


def test_all_functions_shape():
    assert positive_count(_DATA).shape == (3,)
    assert positive_std(_DATA).shape == (3,)
    assert positive_percentile(_DATA, q=25).shape == (3,)
    assert positive_se(_DATA).shape == (3,)
