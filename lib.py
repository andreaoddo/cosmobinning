from __future__ import annotations
from numpy import ndarray, int32
from numpy import arange, linspace, digitize, bincount, meshgrid, fromiter, flip
from numpy import sqrt as npsqrt
from typing import Callable
from math import isqrt, isclose


class Bin:
	def __init__(self, inf: float, sup: float):
		self.inf = inf
		self.sup = sup

	def __repr__(self) -> str:
		return f"[{self.inf},{self.sup})"

	def __eq__(self, other: Bin) -> bool:
		return isclose(self.inf, other.inf) and isclose(self.sup, other.sup)

	def center(self):
		return 0.5 * (self.inf + self.sup)


class Bins:
	def __init__(self, bins: list[Bin]):
		for (i, b) in enumerate(bins[:-1]):
			assert bins[i].sup == bins[i + 1].inf
		self.bins = bins

	def __repr__(self):
		return self.bins.__repr__()

	def centers(self) -> list[float]:
		return list(map(lambda b: b.center(), self.bins))

	@staticmethod
	def linear_bins(first_center: float, width: float, bins: int) -> Bins:
		return Bins([Bin(first_center + (i - 0.5) * width, first_center + (i + 0.5) * width) for i in range(bins)])

	def edges(self) -> list[float]:
		return [b.inf for b in self.bins] + [self.bins[-1].sup]

	def int_max(self) -> int:
		return int(max(map(lambda b: b.sup, self.bins)))

	def squared_max(self) -> int:
		exact_max = max(map(lambda b: b.sup, self.bins))
		return int((exact_max * (1 - 1.e-10)) ** 2)

	def square_roots_range(self) -> ndarray:
		return npsqrt(arange(self.squared_max() + 1))

	def bin_positions(self) -> ndarray:
		return digitize(self.square_roots_range(), self.edges())

	def mode_counts(self) -> ndarray:
		int_max = self.int_max()
		grid_1d = linspace(-int_max, int_max, 2 * int_max + 1, dtype=int32)

		x2, y2 = meshgrid(grid_1d ** 2, grid_1d ** 2, sparse=True)
		count_grid_2d = bincount((x2 + y2).flatten())
		squares = linspace(0, int_max, int_max + 1, dtype=int32) ** 2

		return fromiter(self.mode_counter_generator(count_grid_2d, squares), dtype=int32)

	def mode_counter_generator(self, count_grid: ndarray, squares: ndarray):
		for q2 in range(self.squared_max() + 1):
			count_flipped = flip(count_grid[:q2 + 1])
			yield 2 * count_flipped[squares[:isqrt(q2) + 1]].sum() - count_flipped[0]


class RealSpacePowerBinner:
	def __init__(self, bins: Bins):
		self.bins = bins
		self.counts = bins.mode_counts()
		self.pos = bins.bin_positions()
		self.zero_pos = (min(self.pos) == 0)
		self.bin_counts = bincount(self.pos, weights=self.counts)
		self.inputs = bins.square_roots_range()

	def bin_function(self, function: Callable[[float | ndarray], float | ndarray]) -> ndarray:
		result = bincount(self.pos, weights=function(self.inputs) * self.counts) / self.bin_counts
		if self.zero_pos:
			return result[1:]
		return result
