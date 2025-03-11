"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_mean

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""  

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_mean_my_val():
    """Test that mean function works for my vals"""
    test_input = np.array([[2,4],
                           [3,6],
                           [4,8]])
    test_result = np.array([3,6])
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_mean_unexpected_type():
    """Should fail if the type of the input is wrong"""
    test_input = np.array(["a","b","c"])
    with pytest.raises(TypeError):
        daily_mean(test_input)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0],[0,0],[0,0]],[0,0]),
        ([[1,2], [3,4], [5,6]],[3,4]),
    ],
)

def test_daily_mean(test,expected):
    """test that mean function works for an array of zeros and positive integers"""
    npt.assert_array_equal(daily_mean(np.array(test)),np.array(expected))