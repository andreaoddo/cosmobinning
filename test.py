import pytest
from cosmobinning.lib import Bins, Bin, RealSpacePowerBinner
from numpy import arange, sqrt, allclose, array


def test_bin_center():
	some_bin = Bin(2, 3.5)
	assert 2.75 == some_bin.center()


def test_bins_centers():
	bin1 = Bin(1, 2)
	bin2 = Bin(2, 3.5)
	bin3 = Bin(3.5, 5.5)

	bins = Bins([bin1, bin2, bin3])

	assert [1.5, 2.75, 4.5] == bins.centers()


def test_linear_bins():
	bins = Bins.linear_bins(1.0, 1.0, 3)

	exp_bins = Bins([
		Bin(0.5, 1.5), Bin(1.5, 2.5), Bin(2.5, 3.5)
	])

	assert exp_bins.bins == bins.bins


def test_edges():
	bins = Bins.linear_bins(0.5, 1, 5)
	exp_edges = [0, 1, 2, 3, 4, 5]

	assert exp_edges == bins.edges()


def test_int_max():
	bins = Bins.linear_bins(1, 1.5, 20)

	assert 30 == bins.int_max()


def test_squared_max():
	bins = Bins.linear_bins(1, 1.5, 20)

	assert 915 == bins.squared_max()


def test_square_roots_range():
	bins = Bins.linear_bins(1, 1, 10)

	assert allclose(sqrt(arange(110 + 1)), bins.square_roots_range())


def test_bin_positions():
	bins = Bins.linear_bins(0.75, 0.5, 5)
	exp_positions = array([0, 2, 2, 3, 4, 4, 4, 5, 5])

	assert allclose(exp_positions, bins.bin_positions())


def test_mode_counts():
	bins = Bins.linear_bins(1, 1, 3)

	exp_counts = [1, 6, 12, 8, 6, 24, 24, 0, 12, 30, 24, 24, 8]

	assert all(exp_counts == bins.mode_counts())


def test_bin_function():
	bins = Bins.linear_bins(1, 1, 5)
	binner = RealSpacePowerBinner(bins)
	x_binned = binner.bin_function(lambda x: x)

	expected = [1.2761424, 2.2308031, 3.1341592, 4.0605798, 5.0975831]

	assert allclose(expected, x_binned)
