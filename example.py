from cosmobinning.lib import Bins, RealSpacePowerBinner, RedshiftSpacePowerBinner
import matplotlib.pyplot as plt
import numpy as np
from time import time


def func(x):
	return x ** 2 - x + 1


def anis_func(x, _mu):
	return x ** 2 - x + 1

bins = Bins.linear_bins(1.0, 1.0, 128)
binner = RealSpacePowerBinner(bins)

x_val = bins.square_roots_range()
y_val = func(x_val)

x_bin = bins.centers()
y_bin = binner.bin_function(func)

x_eff = binner.bin_function(lambda x: x)
y_eff = func(x_eff)

z_binner = RedshiftSpacePowerBinner(bins)
y_rsd = z_binner.bin_function(anis_func, 0)

plt.semilogy(x_bin, np.abs(1 - y_eff / y_bin), 'o')
plt.semilogy(x_bin, np.abs(1 - y_rsd / y_bin), 'o')
plt.xlabel('x')
plt.ylabel('Relative error to real space binned')
plt.show()
