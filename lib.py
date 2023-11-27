from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from warnings import warn
from time import sleep, time

from numpy import ndarray, int16, int32, float64, arange, linspace, digitize, bincount, meshgrid, fromiter, flip, fromfunction, zeros, divide, array, absolute, logspace, loadtxt
from numpy import sqrt as npsqrt
from numpy import sum as npsum
from typing import Callable, Optional, Tuple
from math import isqrt, isclose, log10, factorial
import pathlib
from scipy.special import eval_legendre
from scipy.misc import derivative
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline
from sys import getsizeof


class Bin:
	def __init__(self, inf: float, sup: float, right_open: bool = True) -> None:
		"""
		Initializes a Bin object instance
		:param inf: Inferior bound of the bin
		:type inf: float
		:param sup: Superior bound of the bin
		:type sup: float
		"""
		self.inf = inf
		self.sup = sup
		self.right_open = right_open

	def __repr__(self) -> str:
		"""
		Returns the string representation of the object
		:return: String representation of the object
		:rtype: str
		"""
		if self.right_open:
			return f"[{self.inf},{self.sup})"
		else:
			return f"({self.inf},{self.sup}]"

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

	def __hash__(self):
		return hash((self.inf, self.sup))

	def center(self) -> float:
		"""
		Returns the central value of the bin.
		:return: The central value of the bin
		:rtype: float
		"""
		return 0.5 * (self.inf + self.sup)

	def contains(self, k: float) -> bool:
		if self.right_open:
			return self.inf <= k < self.sup
		else:
			return self.inf < k <= self.sup


class Bins:
	def __init__(self, bins: list[Bin]) -> None:
		"""
		Initializes a Bins instance object
		:param bins: A list of Bin objects
		:type bins: list[Bin]
		:param right_open: Indicates whether bins are right-open or not, defaults to True
		:type right_open: bool
		"""

		assert all(map(lambda b: b.right_open, bins)) or all(map(lambda b: not b.right_open, bins))

		for (i, b) in enumerate(bins[:-1]):
			assert bins[i].sup == bins[i + 1].inf
		self.bins = bins
		self.right_open = bins[0].right_open

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
	def linear_bins(binning_scheme: BinningScheme) -> Bins:
		"""
		Constructor static method to create a Bins instance object with linearly spaced bins with equal width
		:param binning_scheme: binning scheme from which the linear bins are created
		:type binning_scheme: BinningScheme
		:return: Bins instance object with linearly spaced bins
		:rtype: Bins
		"""
		return Bins([Bin(binning_scheme.first_center + (i - 0.5) * binning_scheme.width, binning_scheme.first_center + (i + 0.5) * binning_scheme.width, right_open=binning_scheme.right_open) for i in range(binning_scheme.bin_count)])

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
		return digitize(self.square_roots_range(), self.edges(), right = not self.right_open)

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


class BinningScheme:
	def __init__(self, first_center: float, width: float, bin_count: int, right_open: bool = True):
		self.first_center = first_center
		self.width = width
		self.bin_count = bin_count
		self.right_open = right_open

	def can_be_rebinned_into(self, other: BinningScheme) -> bool:
		input_bins = Bins.linear_bins(self)
		output_bins = Bins.linear_bins(other)

		if self.right_open != other.right_open:
			return False

		width_ratio = other.width/self.width
		if not width_ratio.is_integer() or width_ratio < 1:
			return False

		if (input_bins.bins[0].inf > output_bins.bins[0].inf) or (input_bins.bins[-1].sup < output_bins.bins[-1].sup):
			return False

		# For each output bin, there must be an input bin sharing the same inf and one sharing the same sup
		for out_b in output_bins.bins:
			check_inf = False
			check_sup = False
			for in_b in input_bins.bins:
				check_inf = check_inf or (in_b.inf == out_b.inf)
				check_sup = check_sup or (in_b.sup == out_b.sup)
			if not(check_inf and check_sup):
				return False

		return True





class PowerBinner:
	"""
	Common builder for all possible implemented Binners; hides details on specific Binner instances
	"""

	@staticmethod
	def new(bins: Bins, method: str, space: str, multipoles: Optional[list[int]] = None) -> Binner:
		"""
		Builds a Binner object specific for the input parameters
		:param bins: Bins object to be used by the Binner
		:type bins: Bins
		:param method: binning method to be used by the Binner, ["average" | "effective" | "expansion"]
		:type method: str
		:param space: whether the Binner is supposed to be for real or redshift space, ["real" | "redshift"]
		:type space: str
		:param multipoles: optional list of multipoles for a redshift-space binner, must not be specified if Space.REAL; defaults to None, corresponding to l = [0,2,4]
		:type multipoles: Optional[list[int]]
		:return: Returns a Binner instance corresponding to the input parameters
		:rtype: Binner
		"""
		if space.lower() == "real":
			assert multipoles is None, "Multipoles don't have to be specified in real space"
			if method.lower() == "average":
				return RealSpacePowerAverageBinner(bins)
			elif method.lower() == "effective":
				return RealSpacePowerEffectiveBinner(bins)
			elif method.lower() == "expansion":
				return RealSpacePowerExpansionBinner(bins)
		elif space.lower() == "redshift":
			if method.lower() == "average":
				return RedshiftSpacePowerAverageBinner(bins, multipoles)
			elif method.lower() == "effective":
				return RedshiftSpacePowerEffectiveBinner(bins, multipoles)
			elif method.lower() == "expansion":
				return RedshiftSpacePowerExpansionBinner(bins, multipoles)

		raise NotImplementedError


class Binner(ABC):
	"""
	Abstract class grouping all Binners
	"""

	@abstractmethod
	def bin_function(self, function: Callable, x_scale: float = 1.0) -> ndarray:
		"""
		Interface method to be implemented by all concrete Binner types. Computes the binning of the given function for the method specified in construction
		:param function: function to be binned
		:type function: Callable
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


class TriangleBin:
	def __init__(self, bin1: Bin, bin2: Bin, bin3: Bin) -> None:
		assert bin1.right_open == bin2.right_open and bin2.right_open == bin3.right_open
		self.bin1 = bin1
		self.bin2 = bin2
		self.bin3 = bin3

	def __hash__(self):
		return hash((self.bin1, self.bin2, self.bin3))

	def __eq__(self, other):
		return self.bin1 == other.bin1 and self.bin2 == other.bin2 and self.bin3 == other.bin3

	def center(self) -> Tuple[float, float, float]:
		return self.bin1.center(), self.bin2.center(), self.bin3.center()

	def contains(self, triangle: Tuple[float, float, float]) -> bool:
		k1, k2, k3 = triangle
		return self.bin1.contains(k1) and self.bin2.contains(k2) and self.bin3.contains(k3)


class TriangleBins:
	def __init__(self, bins: list[TriangleBin], right_open: bool = True, open_triangles = True):
		self.bins = bins
		self.right_open = right_open
		self.open_triangles = open_triangles

	@staticmethod
	def linear_bins(binning_scheme: BinningScheme, open_triangles: bool = True) -> TriangleBins:
		one_d_bins = Bins.linear_bins(binning_scheme)
		lin_bins = []

		for i1 in range(binning_scheme.bin_count):
			for i2 in range(i1+1):
				for i3 in range(i2+1):
					k1 = one_d_bins.bins[i1]
					k2 = one_d_bins.bins[i2]
					k3 = one_d_bins.bins[i3]

					if k2.sup + k3.sup > k1.inf:
						if open_triangles or k2.center() + k3.center() >= k1.center():
							lin_bins.append(TriangleBin(k1, k2, k3))

		return TriangleBins(lin_bins, binning_scheme.right_open, open_triangles)


class OrderedTriangle:
	def __init__(self, a: float, b: float, c: float):
		self.sup = max(a, max(b, c))
		self.inf = min(a, min(b, c))
		self.med = a + b + c - self.sup - self.inf

	def __add__(self, other: OrderedTriangle) -> OrderedTriangle:
		return OrderedTriangle(self.sup+other.sup, self.med+other.med, self.inf+other.inf)

	def __mul__(self, other: float) -> OrderedTriangle:
		assert other > 0
		return OrderedTriangle(other*self.sup, other*self.med, other*self.inf)

	def __rmul__(self, other):
		assert other > 0
		return OrderedTriangle(other*self.sup, other*self.med, other*self.inf)



class RealSpaceBispectrumEffectiveBinner(Binner):
	def __init__(self, bins: TriangleBins, effective_triangles: list[OrderedTriangle], counts: list[int]):
		self.bins = bins
		self.effective_triangles = effective_triangles
		self.counts = counts

	def bin_function(self, function: Callable, x_scale: float = 1.0) -> ndarray:
		return array([function(t.sup*x_scale, t.med*x_scale, t.inf*x_scale) for t in self.effective_triangles])

	@staticmethod
	def new(binning_scheme: BinningScheme, open_triangles: bool = True) -> RealSpaceBispectrumEffectiveBinner:

		available_schemes = [
			(BinningScheme(1, 1, 64), "effectives/effective_3D_c1.00_dk1.00_n64_leftopen_opentri-true.txt"),
			(BinningScheme(1.5, 1, 64, right_open=True), "effectives/effective_3D_c1.50_dk1.00_n63_rightopen_opentri-true.txt"),
			(BinningScheme(0.5, 1, 64, right_open=False), "effectives/effective_3D_c0.50_dk1.00_n64_leftopen_opentri-true.txt")
		]

		for from_scheme, file_path in available_schemes:
			if from_scheme.can_be_rebinned_into(binning_scheme):
				binner = RealSpaceBispectrumEffectiveBinner._from_file(pathlib.Path(__file__).parent.resolve().joinpath(file_path), from_scheme)
				return binner._rebin(from_scheme, binning_scheme, open_triangles=open_triangles)

		raise InvalidBispectrumBinningSchemeException

	@staticmethod
	def _from_file(file_path: str, binning_scheme: BinningScheme) -> RealSpaceBispectrumEffectiveBinner:
		triangle_bins = TriangleBins.linear_bins(binning_scheme)
		center_1, center_2, center_3, k1_effective, k2_effective, k3_effective, count = loadtxt(file_path, unpack=True)
		effective_triangles = []
		counts = []
		for i, bin in enumerate(triangle_bins.bins):
			assert bin.center() == (center_1[i], center_2[i], center_3[i]), "The file does not contain triangles for the given binning scheme"
			effective_triangles.append(OrderedTriangle(k1_effective[i], k2_effective[i], k3_effective[i]))
			counts.append(count[i])

		binner = RealSpaceBispectrumEffectiveBinner(triangle_bins, effective_triangles, counts)
		return binner

	def _rebin(self, from_scheme: BinningScheme, to_scheme: BinningScheme, open_triangles: bool = True) -> RealSpaceBispectrumEffectiveBinner:
		if not from_scheme.can_be_rebinned_into(to_scheme):
			raise InvalidBispectrumBinningSchemeException

		new_bins = TriangleBins.linear_bins(to_scheme, open_triangles=open_triangles)
		new_effective = [OrderedTriangle(0, 0, 0)] * len(new_bins.bins)
		new_counts = [0] * len(new_bins.bins)
		new_binner = RealSpaceBispectrumEffectiveBinner(new_bins, new_effective, new_counts)

		input_linear_bins = Bins.linear_bins(from_scheme)
		output_linear_bins = Bins.linear_bins(to_scheme)

		mapping = { b: next((c for c in output_linear_bins.bins if c.contains(b.center())), None) for b in input_linear_bins.bins }
		triangle_mapping = {}
		for triangle_input_bin in self.bins.bins:
			b1 = mapping[triangle_input_bin.bin1]
			b2 = mapping[triangle_input_bin.bin2]
			b3 = mapping[triangle_input_bin.bin3]
			if b1 is not None and b2 is not None and b3 is not None:
				triangle_mapping[triangle_input_bin] = TriangleBin(b1, b2, b3)
			else:
				triangle_mapping[triangle_input_bin] = None
		bins_to_index_mapping = { output_triangle_bin: i for i, output_triangle_bin in enumerate(new_bins.bins) }

		for j, input_triangle_bin in enumerate(self.bins.bins):
			output_triangle_bin = triangle_mapping.get(input_triangle_bin)
			if output_triangle_bin is not None:
				idx = bins_to_index_mapping.get(output_triangle_bin)
				if idx is not None:
					print(j)
					count = self.counts[j]
					new_counts[idx] += count
					new_effective[idx] += count * self.effective_triangles[j]

		for i in range(len(new_binner.bins.bins)):
			new_effective[i] = new_effective[i] * (1/new_counts[i])
			print(new_bins.bins[i].bin1, new_bins.bins[i].bin2, new_bins.bins[i].bin3, new_effective[i].sup, new_effective[i].med, new_effective[i].inf, new_counts[i])
		new_binner.effective_triangles = new_effective
		new_binner.counts = new_counts
		return new_binner

class InvalidBispectrumBinningSchemeException(Exception):
	"The binning scheme selected for the bispectrum is invalid. Only schemes that can be remapped from BinningScheme(0.5, 1, 64) or BinningScheme(1, 1, 64) are valid."
	pass