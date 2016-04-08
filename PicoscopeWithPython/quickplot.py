import numpy as np
import matplotlib.pyplot as plt
import glob as gb

files = gb.glob('./Data/*.txt')
# files = sorted(files, key=lambda x: int(x.split('\\')[1][0:-4]))
# Import colorbar
import matplotlib.cm as mplcm
import matplotlib.colors as colors
NUM_COLORS = len(files)
cm = plt.get_cmap('rainbow')
cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
from cycler import cycler

# Plot data
fig = plt.figure(figsize=(10.0,5.0))
ax = fig.add_subplot(111)
ax.set_prop_cycle(cycler('color', [scalarMap.to_rgba(i) for i in range(NUM_COLORS)]))

for f in files:
    data = np.genfromtxt(fname=f, delimiter=' ', dtype=float)
    y = data[:, 1]
    t = data[:,0]
    y = y - min(y)
    y = y / max(y)
    ax.plot(t,y, label=f.split('\\')[1][0:-4])

# ax.axhline(1/np.e, color='k', linestyle='--', label='1/e')
plt.legend(loc='best', title='Reflectance %', prop={'size': 14}, ncol=2)
plt.xlabel('Time (ms)')
plt.ylabel('Intensity (A.U.)')
ax.set_yscale('log')
# plt.xlim([0,100])
plt.ylim([0.01, 1])
plt.savefig('Data/picoscope_raw_data.png', dpi=1000)
plt.show()