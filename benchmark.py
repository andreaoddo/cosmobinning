from cosmobinning.lib import Bins, Bin, RealSpacePowerBinner, RedshiftSpacePowerBinner
from timeit import timeit
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sys import getsizeof

k, Pk = np.loadtxt(
	"/media/andrea/DATA/BackToSchool/BinningML/cosmo-binning/cosmobinning/leading_order_power_spectrum_Minerva_z1.txt",
	unpack=True)
k = np.insert(k, 0, 0)
Pk = np.insert(Pk, 0, 0)
kF = 2 * np.pi / 1500.
PkL = interp1d(k / kF, Pk, kind='cubic')


def real_space_power_spectrum(k):
	return PkL(k)


def redshift_space_power_spectrum(k, mu):
	return (1 + 0.8 * mu ** 2) ** 2 * PkL(k)


number = 100

real_space_times = []
real_space_bins = [16, 32, 64, 128, 256, 512, 1024]
real_space_mem = []
for n in real_space_bins:
	binner = RealSpacePowerBinner(Bins.linear_bins(1, 1, n))


	def bench_real_space():
		binner.bin_function(real_space_power_spectrum)
		pass

	avg_time = timeit(bench_real_space, number=number) / number
	real_space_times.append(avg_time)
	real_space_mem.append(getsizeof(binner) / (1 << 20))


redshift_space_times = []
redshift_space_mem = []
number = 10
redshift_space_bins = [16, 32, 64, 128, 256]
for n in redshift_space_bins:
	binner = RedshiftSpacePowerBinner(Bins.linear_bins(1, 1, n))


	def bench_redshift_space():
		binner.bin_function(redshift_space_power_spectrum)
		pass


	avg_time = timeit(bench_redshift_space, number=number) / number
	redshift_space_times.append(avg_time)
	redshift_space_mem.append(getsizeof(binner) / (1 << 20))


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.loglog(real_space_bins, real_space_times, label='real space', color='tab:blue')
ax1.loglog(redshift_space_bins, redshift_space_times, label='redshift space (3 multipoles)', color='tab:red')
for n in real_space_bins:
	ax1.axvline(n, ls='--', lw=0.8, color='tab:gray')
ax1.set_xlabel('Number of bins (first center = 1, width = 1)')
ax1.set_ylabel('Time [s] (continuous)')
ax1.legend(loc=0)

ax2.loglog(real_space_bins, real_space_mem, color='tab:blue', ls='--')
ax2.loglog(redshift_space_bins, redshift_space_mem, color='tab:red', ls='--')
ax2.set_ylabel('Estimated memory allocation [MB] (dashed)')
plt.savefig("benchmark.png", bbox_inches='tight', dpi=300)
