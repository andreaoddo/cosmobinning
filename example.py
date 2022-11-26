from cosmobinning.lib import Bins, RealSpacePowerBinner
import matplotlib.pyplot as plt
import numpy as np


def func(x):
	return x ** 2 - x + 1


bins = Bins.linear_bins(1.0, 1.0, 128)
binner = RealSpacePowerBinner(bins)

x_val = bins.square_roots_range()
y_val = func(x_val)

x_bin = bins.centers()
y_bin = binner.bin_function(func)

x_eff = binner.bin_function(lambda x: x)
y_eff = func(x_eff)

plt.plot(x_bin, 100 * np.abs(1 - y_eff / y_bin), 'o')
plt.xlabel('x')
plt.ylabel('Relative error of effective [%]')
plt.show()
