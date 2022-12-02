from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from warnings import warn
from time import sleep

from numpy import ndarray, int16, int32, float64, arange, linspace, digitize, bincount, meshgrid, fromiter, flip, fromfunction, zeros, divide, array, absolute, logspace
from numpy import sqrt as npsqrt
from numpy import sum as npsum
from typing import Callable, Optional
from math import isqrt, isclose, log10, factorial
from scipy.special import eval_legendre
from scipy.misc import derivative
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline
from sys import getsizeof


class Bin:
	def __init__(self, inf: float, sup: float) -> None:
		"""
		Initializes a Bin object instance
		:param inf: Inferior bound of the bin
		:type inf: float
		:param sup: Superior bound of the bin
		:type sup: float
		"""
		self.inf = inf
		self.sup = sup

	def __repr__(self) -> str:
		"""
		Returns the string representation of the object
		:return: String representation of the object
		:rtype: str
		"""
		return f"[{self.inf},{self.sup})"

	def __eq__(self, other: Bin) -> bool:
		"""
		Checks if the instance object is equal to the other given object
		:param other: Bin instance to check equality with
		:return: Equality check
		:rtype: bool
		"""
		return isclose(self.inf, other.inf) and isclose(self.sup, other.sup)

	def __sizeof__(self) -> int:
		"""
		Returns the size of the object in bytes
		:return: Size of the object in bytes
		:rtype: int
		"""
		return getsizeof(self.inf) + getsizeof(self.sup)

	def center(self) -> float:
		"""
		Returns the central value of the bin.
		:return: The central value of the bin
		:rtype: float
		"""
		return 0.5 * (self.inf + self.sup)


class Bins:
	def __init__(self, bins: list[Bin]) -> None:
		"""
		Initializes a Bins instance object
		:param bins: A list of Bin objects
		:type bins: list[Bin]
		"""
		for (i, b) in enumerate(bins[:-1]):
			assert bins[i].sup == bins[i + 1].inf
		self.bins = bins

	def __repr__(self):
		"""
		Returns the string representation of the object
		:return: String representation of the object
		:rtype: str
		"""
		return self.bins.__repr__()

	def __sizeof__(self) -> int:
		"""
		Returns the size of the object in bytes
		:return: Size of the object in bytes
		:rtype: int
		"""
		return getsizeof(self.bins)

	def centers(self) -> list[float]:
		"""
		Returns a list of the central values of each bin in the instance object
		:return: List of central value
		:rtype: list[float]
		"""
		return list(map(lambda b: b.center(), self.bins))

	@staticmethod
	def linear_bins(first_center: float, width: float, bins: int) -> Bins:
		"""
		Constructor static method to create a Bins instance object with linearly spaced bins with equal width
		:param first_center: Center of the first bin
		:type first_center: float
		:param width: Width of each bin
		:type width: float
		:param bins: Number of bins
		:type bins: int
		:return: Bins instance object with linearly spaced bins
		:rtype: Bins
		"""
		return Bins([Bin(first_center + (i - 0.5) * width, first_center + (i + 0.5) * width) for i in range(bins)])

	def edges(self) -> list[float]:
		"""
		Returns a list of the edges of the bins.
		:return: List of bin edges
		:rtype: list[float]
		"""
		return [b.inf for b in self.bins] + [self.bins[-1].sup]

	def grid_size(self) -> int:
		"""
		Returns the minimum grid size that defines the Bins
		:return: Grid size
		:rtype: int
		"""
		return int(max(map(lambda b: b.sup, self.bins)))

	def squared_max(self) -> int:
		"""
		Returns the maximum distance squared defined by the bins
		:return: Maximum distance squared
		:rtype: int
		"""
		exact_max = max(map(lambda b: b.sup, self.bins))
		return int((exact_max * (1 - 1.e-16)) ** 2)

	def square_roots_range(self) -> ndarray:
		"""
		Returns all square roots of numbers in the range 0 to squared_max()
		:return: Square roots of numbers
		:rtype: ndarray
		"""
		return npsqrt(arange(self.squared_max() + 1))

	def bin_positions(self) -> ndarray:
		"""
		Returns a list of integers representing the bin in which each value in square_roots_range() belongs.
		:return: bin positions of all values in square_roots_range()
		:rtype: ndarray
		"""
		return digitize(self.square_roots_range(), self.edges())

	def mode_counts_3d(self) -> ndarray:
		"""
		For each possible square distance, counts the number of points on a 3D grid having such squared distance, and returns the number of points in a list
		:return: The list of grid points counts
		:rtype: ndarray
		"""
		int_max = self.grid_size()
		grid_1d = linspace(-int_max, int_max, 2 * int_max + 1, dtype=int32)

		x2, y2 = meshgrid(grid_1d ** 2, grid_1d ** 2, sparse=True)
		count_grid_2d = bincount((x2 + y2).flatten())
		squares = linspace(0, int_max, int_max + 1, dtype=int32) ** 2

		return fromiter(self.mode_counter_generator(count_grid_2d, squares), dtype=int32)

	def mode_counts_2d(self) -> ndarray:
		"""
		For each possible square distance, counts the number of points on a 2D grid having such squared distance, and
		returns the number of points in a list
		:return: The list of grid points counts
		:rtype: ndarray
		"""
		int_max = self.grid_size()
		grid_1d = linspace(-int_max - 1, int_max + 1, 2 * int_max + 3, dtype=int32)

		x2 = grid_1d ** 2
		count_grid_2d = bincount(x2)
		squares = linspace(0, int_max, int_max + 1, dtype=int32) ** 2

		return fromiter(self.mode_counter_generator(count_grid_2d, squares), dtype=int16)

	def mode_counter_generator(self, count_grid: ndarray, squares: ndarray):
		"""
		Generator for the number of points on grids
		:param count_grid:
		:param squares: List of square numbers
		"""
		for q2 in range(self.squared_max() + 1):
			count_flipped = flip(count_grid[:q2 + 1])
			yield 2 * count_flipped[squares[:isqrt(q2) + 1]].sum() - count_flipped[0]


class BinningMethod(Enum):
	"""
	Enum-like class enumerating all possible binning methods supported
	"""
	AVERAGE = auto()
	EFFECTIVE = auto()
	EXPANSION = auto()


class Space(Enum):
	"""
	Enum-like class representing either real-space or redshift-space
	"""
	REAL = auto()
	REDSHIFT = auto()


class BinnerFactory:
	"""
	Common builder for all possible implemented Binners; hides details on specific Binner instances
	"""

	@staticmethod
	def build(bins: Bins, method: BinningMethod, space: Space, multipoles: Optional[list[int]] = None) -> Binner:
		"""
		Builds a Binner object specific for the input parameters
		:param bins: Bins object to be used by the Binner
		:type bins: Bins
		:param method: binning method to be used by the Binner
		:type method: BinningMethod
		:param space: whether the Binner is supposed to be for real or redshift space
		:type space: Space
		:param multipoles: optional list of multipoles for a redshift-space binner, must not be specified if Space.REAL; defaults to None, corresponding to l = [0,2,4]
		:type multipoles: Optional[list[int]]
		:return: Returns a Binner instance corresponding to the input parameters
		:rtype: Binner
		"""
		if space is Space.REAL:
			assert multipoles is None, "Multipoles don't have to be specified in real space"
			if method is BinningMethod.AVERAGE:
				return RealSpacePowerAverageBinner(bins)
			elif method is BinningMethod.EFFECTIVE:
				return RealSpacePowerEffectiveBinner(bins)
			elif method is BinningMethod.EXPANSION:
				return RealSpacePowerExpansionBinner(bins)
		elif space is Space.REDSHIFT:
			if method is BinningMethod.AVERAGE:
				return RedshiftSpacePowerAverageBinner(bins, multipoles)
			elif method is BinningMethod.EFFECTIVE:
				return RedshiftSpacePowerEffectiveBinner(bins, multipoles)
			elif method is BinningMethod.EXPANSION:
				return RedshiftSpacePowerExpansionBinner(bins, multipoles)

		raise NotImplementedError


class Binner(ABC):
	"""
	Abstract class grouping all Binners
	"""

	@abstractmethod
	def bin_function(self, power: Callable, x_scale: float = 1.0) -> ndarray:
		"""
		Interface method to be implemented by all concrete Binner types. Computes the binning of the given function for the method specified in construction
		:param power: function to be binned
		:type power: Callable
		:param x_scale: scale on the x-axis (usually equals the fundamental frequency), defaults to 1
		:type x_scale: float
		"""
		raise NotImplementedError


class RealSpacePowerAverageBinner(Binner):
	def __init__(self, bins: Bins) -> None:
		"""
		Initializes an instance of a RealSpacePowerAverageBinner
		:param bins: bins over which the binning average is to be computed
		:type bins: Bins
		"""
		self.bins = bins
		self.counts = bins.mode_counts_3d()
		self.pos = bins.bin_positions()
		self.zero_pos = (min(self.pos) == 0)
		self.bin_counts = bincount(self.pos, weights=self.counts)
		self.inputs = bins.square_roots_range()

	def __sizeof__(self) -> int:
		"""
		Returns the size of the object in bytes
		:return: Size of the object in bytes
		:rtype: int
		"""
		return getsizeof(self.bins) + self.counts.nbytes + self.pos.nbytes + getsizeof(
			True) + self.bin_counts.nbytes + self.inputs.nbytes

	def bin_function(self, power: Callable[[float | ndarray], float | ndarray], x_scale: float = 1.0) -> ndarray:
		"""
		Computes the bin average of the given isotropic function
		:param power: Real function with domain in |R
		:type power: Callable[[float | ndarray], float | ndarray]
		:param x_scale: scale on the x-axis (usually equals the fundamental frequency), defaults to 1
		:type x_scale: float
		:return: Bin average of the given function
		:rtype: ndarray
		"""
		result = bincount(self.pos, weights=power(self.inputs*x_scale) * self.counts) / self.bin_counts
		if self.zero_pos:
			return result[1:]
		return result


class RealSpacePowerEffectiveBinner(Binner):

	def __init__(self, bins: Bins):
		"""
		Initializes an instance of a RealSpacePowerEffectiveBinner
		:param bins: bins over which the binning average is to be computed
		:type bins: Bins
		"""
		self.bins = bins
		binner = RealSpacePowerAverageBinner(bins)
		self.k_effective = binner.bin_function(lambda k: k)

		counts = bins.mode_counts_3d()
		pos = bins.bin_positions()
		self.bin_counts = bincount(pos, weights=counts)

	def bin_function(self, power: Callable[[float | ndarray], float | ndarray], x_scale: float = 1.0) -> ndarray:
		"""
		Computes the bin average of the given isotropic function, using the Effective method
		:param power: Real function with domain in |R
		:type power: Callable[[float | ndarray], float | ndarray]
		:param x_scale: scale on the x-axis (usually equals the fundamental frequency), defaults to 1
		:type x_scale: float
		:return: Bin average of the given function
		:rtype: ndarray
		"""
		return power(self.k_effective*x_scale)


class RealSpacePowerExpansionBinner(Binner):

	def __init__(self, bins: Bins):
		"""
		Initializes an instance of a RealSpacePowerExpansionBinner
		:param bins: bins over which the binning average is to be computed
		:type bins: Bins
		"""
		self.bins = bins
		binner = RealSpacePowerAverageBinner(bins)
		self.k_eff = binner.bin_function(lambda k: k)
		self.k_eff2 = binner.bin_function(lambda k: k ** 2)
		self.mom_2 = self.k_eff2 - self.k_eff ** 2

		counts = bins.mode_counts_3d()
		pos = bins.bin_positions()
		self.bin_counts = bincount(pos, weights=counts)

	def bin_function(self, power: Callable[[float | ndarray], float | ndarray], x_scale: float = 1.0) -> ndarray:
		"""
		Computes the bin average of the given isotropic function, using the Effective method
		:param power: Real function with domain in |R
		:type power: Callable[[float | ndarray], float | ndarray]
		:param x_scale: scale on the x-axis (usually equals the fundamental frequency), defaults to 1
		:type x_scale: float
		:return: Bin average of the given function
		:rtype: ndarray
		"""
		return (power(self.k_eff*x_scale) + fromiter(
			map(lambda k: derivative(power, k, dx=1.e-4, n=2), self.k_eff*x_scale), dtype=float64) * self.mom_2 * x_scale ** 2 / 2)


class RedshiftSpacePowerAverageBinner(Binner):
	def __init__(self, bins: Bins, multipoles: Optional[list[int]] = None):
		"""
		Initializes an instance of a RedshiftSpacePowerAverageBinner
		:param bins: bins over which the binning average is to be computed
		:type bins: Bins
		:param multipoles: list of multipoles for which the bin average is going to be computed; defaults to None, corresponding to [0, 2, 4]
		:type multipoles: Optional[list[int]]
		"""
		self.bins = bins
		self.counts = bins.mode_counts_2d()
		self.pos = bins.bin_positions()
		self.zero_pos = (min(self.pos) == 0)
		self.bin_counts = bincount(self.pos, weights=bins.mode_counts_3d())
		self.inputs_squared = arange(bins.squared_max() + 1)
		self.inputs = npsqrt(self.inputs_squared)

		self.z_values = arange(-bins.grid_size(), bins.grid_size() + 1)
		if multipoles is None:
			multipoles = [0, 2, 4]
		self.multipoles = multipoles
		arr_multipoles = array(multipoles)[:, None, None]

		mem_to_be_allocated = len(self.z_values) * len(self.inputs) * len(self.multipoles) * 8 / (1 << 30)
		if mem_to_be_allocated >= 1:
			message = f"{mem_to_be_allocated:.2f} GB of memory are about to be allocated\nExecution will continue in 5 seconds"
			warn(message)
			sleep(5)

		cos_theta = zeros([len(self.inputs_squared), 2 * bins.grid_size() + 1])
		divide(self.z_values[None, :], self.inputs[:, None], out=cos_theta, where=(self.inputs_squared[:, None] != 0))

		self.masked_legendre_times_counts = \
			eval_legendre(arr_multipoles, cos_theta[None, :], dtype=float64) * \
			(absolute(self.z_values[None, :]) <= self.inputs[:, None]) * \
			fromfunction(lambda i, j: self.counts[i - (j - bins.grid_size()) ** 2], (len(self.inputs), len(self.z_values)), dtype=int16)
		del cos_theta

	def __sizeof__(self) -> int:
		"""
		Returns the size of the object in bytes
		:return: Size of the object in bytes
		:rtype: int
		"""
		return getsizeof(self.bins) + self.counts.nbytes + self.pos.nbytes + getsizeof(True) + self.bin_counts.nbytes + self.inputs_squared.nbytes + self.inputs.nbytes + self.z_values.nbytes + getsizeof(self.multipoles) + self.masked_legendre_times_counts.nbytes

	def bin_function(self, power: Callable[[ndarray, ndarray], ndarray], x_scale: float = 1.0) -> list[ndarray]:
		"""
		Computes the bin average of the multipoles of the given anisotropic function
		:param power: Real function with domain in |R², (k, mu)
		:type power: Callable[[ndarray, ndarray], ndarray]
		:param x_scale: scale on the x-axis (usually equals the fundamental frequency), defaults to 1
		:type x_scale: float
		:return: Bin average of the multipoles of the given function
		:rtype: ndarray
		"""
		weights = npsum(self.masked_legendre_times_counts * power(self.inputs[:, None]*x_scale, divide(self.z_values[None, :], self.inputs[:, None], where=(self.inputs_squared[:, None] != 0)))[None, :], axis=2)

		result = [(2 * l + 1) * bincount(self.pos, weights=w) / self.bin_counts for (l, w) in zip(self.multipoles, weights)]

		if self.zero_pos:
			for (i, b) in enumerate(result):
				result[i] = b[1:]
			return result
		return result


class RedshiftSpacePowerEffectiveBinner(Binner):

	def __init__(self, bins: Bins, multipoles: Optional[list[int]] = None):
		"""
		Initializes an instance of a RedshiftSpacePowerEffectiveBinner
		:param bins: bins over which the binning average is to be computed
		:type bins: Bins
		:param multipoles: list of multipoles for which the bin average is going to be computed; defaults to None, corresponding to [0, 2, 4]
		:type multipoles: Optional[list[int]]
		"""
		self.bins = bins
		if multipoles is None:
			multipoles = [0, 2, 4]
		self.multipoles = array(multipoles, dtype=int32)

		binner = RealSpacePowerAverageBinner(bins)
		self.effective_k = binner.bin_function(lambda k: k)

		self.pseudo_k = logspace(0, log10(bins.bins[-1].sup), 501)
		self.pseudo_mu = linspace(-1, 1, 51)
		self.legendre = eval_legendre(self.multipoles[:, None, None], self.pseudo_mu[None, None, :])

	def bin_function(self, power: Callable, x_scale: float = 1.0) -> ndarray:
		"""
		Computes the bin average of the multipoles of the given anisotropic function using the Effective method
		:param power: Real function with domain in |R², (k, mu)
		:type power: Callable[[ndarray, ndarray], ndarray]
		:param x_scale: scale on the x-axis (usually equals the fundamental frequency), defaults to 1
		:type x_scale: float
		:return: Bin average of the multipoles of the given function
		:rtype: ndarray
		"""
		function = power(self.pseudo_k[None, :, None]*x_scale, self.pseudo_mu[None, None, :])
		f_l = simpson(self.legendre * function, self.pseudo_mu, axis=-1)
		return array([(l + 0.5) * InterpolatedUnivariateSpline(self.pseudo_k, f_l[i], k=3)(self.effective_k) for (i, l) in enumerate(self.multipoles)])


class RedshiftSpacePowerExpansionBinner(Binner):
	def __init__(self, bins: Bins, multipoles: Optional[list[int]] = None):
		"""
		Initializes an instance of a RedshiftSpacePowerExpansionBinner
		:param bins: bins over which the binning average is to be computed
		:type bins: Bins
		:param multipoles: list of multipoles for which the bin average is going to be computed; defaults to None, corresponding to [0, 2, 4]
		:type multipoles: Optional[list[int]]
		"""
		self.bins = bins
		if multipoles is None:
			multipoles = [0, 2, 4]
		self.multipoles = array(multipoles, dtype=int32)
		self.internal_multipoles = array(list(range(5)), dtype=int32)

		binner = RedshiftSpacePowerAverageBinner(bins, multipoles)
		self.effective_k = binner.bin_function(lambda k, mu: k)[0]

		self.pi_dji = zeros([3, len(self.internal_multipoles), len(self.multipoles), len(self.bins.bins)])
		for (j, el_prime) in enumerate(self.internal_multipoles):
			for d in range(3):
				self.pi_dji[d, j] = array(binner.bin_function(lambda k, mu: k ** d * eval_legendre(el_prime, mu))) / (2 * self.multipoles + 1)[:, None]

		self.pi_dji[2] += self.pi_dji[0] * self.effective_k[None, None, :] ** 2 - 2 * self.effective_k[None, None, :] * self.pi_dji[1]
		self.pi_dji[1] -= self.pi_dji[0] * self.effective_k[None, None, :]

		self.pseudo_k = logspace(0, log10(bins.bins[-1].sup), 501)
		self.pseudo_mu = linspace(-1, 1, 51)
		self.legendre = eval_legendre(self.internal_multipoles[:, None, None], self.pseudo_mu[None, None, :])

	def bin_function(self, power: Callable, x_scale: float = 1.0) -> ndarray:
		"""
		Computes the bin average of the multipoles of the given anisotropic function using the Expansion method
		:param power: Real function with domain in |R², (k, mu)
		:type power: Callable[[ndarray, ndarray], ndarray]
		:param x_scale: scale on the x-axis (usually equals the fundamental frequency), defaults to 1
		:type x_scale: float
		:return: Bin average of the multipoles of the given function
		:rtype: ndarray
		"""
		function = power(self.pseudo_k[None, :, None]*x_scale, self.pseudo_mu[None, None, :])
		power_l = simpson(self.legendre * function, self.pseudo_mu, axis=-1)
		power_q_l = [InterpolatedUnivariateSpline(self.pseudo_k, (l + 0.5) * power_l[i], k=3) for (i, l) in enumerate(self.internal_multipoles)]

		power_eff_l = array([[f.derivative(d)(self.effective_k) / factorial(d) for f in power_q_l] for d in range(3)])

		# power_eff_l has shape (der x inner-mult x bins)
		# pi_dji has shape (der x inner-mult x ext-mult x bins)
		return (2 * self.multipoles + 1)[:, None] * npsum(self.pi_dji * power_eff_l[:, :, None, :], axis=(0, 1))
