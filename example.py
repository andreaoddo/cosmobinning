from cosmobinning.lib import Bins, RealSpacePowerBinner, RedshiftSpacePowerBinner
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

k, Pk = np.loadtxt(
	"/media/andrea/DATA/BackToSchool/BinningML/cosmo-binning/cosmobinning/leading_order_power_spectrum_Minerva_z1.txt",
	unpack=True)
k = np.insert(k, 0, 0)
Pk = np.insert(Pk, 0, 0)
kF = 2 * np.pi / 1500
PkL = interp1d(k / kF, Pk, kind='cubic')
f = 0.8617
b1 = 2.71


def real_space_power_spectrum(k):
	return b1 ** 2 * PkL(k)


def redshift_space_power_spectrum(k, mu):
	return (b1 + f * mu ** 2) ** 2 * PkL(k)


bins = Bins.linear_bins(1.0, 1.0, 128)
binner = RealSpacePowerBinner(bins)

x_val = bins.square_roots_range()
y_val = real_space_power_spectrum(x_val)

x_bin = np.array(bins.centers())
y_bin = binner.bin_function(real_space_power_spectrum)

x_eff = binner.bin_function(lambda x: x)
y_eff = real_space_power_spectrum(x_eff)

z_binner = RedshiftSpacePowerBinner(bins)

Pk0, Pk2, Pk4 = z_binner.bin_function(redshift_space_power_spectrum)

plt.semilogy(x_bin, np.abs(1 - y_eff / y_bin), 'o')
plt.xlabel('x')
plt.ylabel('Peff/Pbin - 1')
plt.show()

plt.plot(x_bin * kF, x_eff * kF * Pk0)
plt.plot(x_bin * kF, x_eff * kF * Pk2)
plt.plot(x_bin * kF, x_eff * kF * Pk4)

plt.show()
