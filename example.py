from cosmobinning.lib import Bins, BinnerFactory, BinningMethod, Space
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

k, Pk = np.loadtxt("cosmobinning/leading_order_power_spectrum_Minerva_z1.txt", unpack=True)
k = np.insert(k, 0, 0)
Pk = np.insert(Pk, 0, 0)
kF = 2 * np.pi / 1500
PkL = interp1d(k, Pk, kind='cubic')
f = 0.8617
b1 = 2.71


def real_space_power_spectrum(k):
	return b1 ** 2 * PkL(k)


def redshift_space_power_spectrum(k, mu):
	return (b1 + f * mu ** 2) ** 2 * PkL(k)


bins = Bins.linear_bins(1.0, 1.0, 128)
binner = BinnerFactory.build(bins, BinningMethod.AVERAGE, Space.REAL)
eff_binner = BinnerFactory.build(bins, BinningMethod.EFFECTIVE, Space.REAL)
exp_binner = BinnerFactory.build(bins, BinningMethod.EXPANSION, Space.REAL)

x_val = bins.square_roots_range()
y_val = real_space_power_spectrum(x_val)

x_bin = np.array(bins.centers())
y_bin = binner.bin_function(real_space_power_spectrum, x_scale=kF)
y_eff = eff_binner.bin_function(real_space_power_spectrum, x_scale=kF)
y_exp = exp_binner.bin_function(real_space_power_spectrum, x_scale=kF)

z_binner = BinnerFactory.build(bins, BinningMethod.AVERAGE, Space.REDSHIFT)
z_binner_eff = BinnerFactory.build(bins, BinningMethod.EFFECTIVE, Space.REDSHIFT)
z_binner_exp = BinnerFactory.build(bins, BinningMethod.EXPANSION, Space.REDSHIFT)

Pk0, Pk2, Pk4 = z_binner.bin_function(redshift_space_power_spectrum, x_scale=kF)
Pk0e, Pk2e, Pk4e = z_binner_eff.bin_function(redshift_space_power_spectrum, x_scale=kF)
Pk0x, Pk2x, Pk4x = z_binner_exp.bin_function(redshift_space_power_spectrum, x_scale=kF)

plt.semilogy(x_bin*kF, np.abs(1 - y_eff / y_bin), 'o', label='Effective')
plt.semilogy(x_bin*kF, np.abs(1 - y_exp / y_bin), 'o', label='Expansion')
plt.xlabel('k')
plt.ylabel('P/Pbin - 1')
plt.legend(loc=0)
plt.show()

x_eff = binner.bin_function(lambda k: k, x_scale=kF)

plt.plot(x_bin * kF, x_eff * Pk0, 'o', color='tab:blue', label='P0 average')
plt.plot(x_bin * kF, x_eff * Pk2, 'o', color='tab:red', label='P2 average')
plt.plot(x_bin * kF, x_eff * Pk4, 'o', color='tab:green', label='P4 average')

plt.plot(x_bin * kF, x_eff * Pk0e, color='tab:blue', ls='--', label='P0 effective')
plt.plot(x_bin * kF, x_eff * Pk2e, color='tab:red', ls='--', label='P2 effective')
plt.plot(x_bin * kF, x_eff * Pk4e, color='tab:green', ls='--', label='P4 effective')

plt.plot(x_bin * kF, x_eff * Pk0x, color='tab:blue', ls='-', label='P0 expansion')
plt.plot(x_bin * kF, x_eff * Pk2x, color='tab:red', ls='-', label='P2 expansion')
plt.plot(x_bin * kF, x_eff * Pk4x, color='tab:green', ls='-', label='P4 expansion')
plt.xlabel('k')
plt.ylabel('P_l')
plt.legend(loc=0)
plt.show()
