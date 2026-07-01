import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 0 deg: 9650 3.90 Hz
# 1 deg: 10075 3.95 Hz
# 2 deg: 10550 4.10 Hz
# 3 deg: 10100 4.33 Hz
# 4 deg:  7250 4.5 Hz 
# 4.5 deg: 7300 4.65 Hz
# 5 deg: 6600 4.75 Hz 
# 6 deg: 2900, 5.2 Hz pitch


"""
finer grid
0. deg: 9250 3.90 Hz
1. deg: 10500 3.90 Hz
2. deg: 11250 > q > 11210 4.3 Hz
3. deg: 10600 < q < 10800 4.5 Hz
4. deg: 9500???
4.5 deg: 7800 4.8 Hz
5. deg: 0.?? 5.1 Hz 
6. deg: 3300

2D fine grid

0. deg: 9450 3.93 Hz
1. deg: 10500 3.90 Hz
2. deg: 11250 > q > 11210 4.3 Hz -> 12000 
3. deg: 10600 < q < 10800 4.5 Hz -> 12750-13000 4.03 Hz 
4. deg: 9500??? 4.6 Hz?
4.5 deg: 7900 4.7 Hz
5. deg: 5250 4.75 Hz 
6. deg: 3300??? 5 Hz 


2D SA options

3. deg: 12400 3.99 Hz 
4. deg: 5500 4.75 Hz 
5. deg: 0. 5.20
6. deg: 0. 5.20 Hz

"""


"""
2D fine grid

AOA = 3 deg 

M = 0.74: 9020
M = 0.76: 
M = 0.78:
M = 0.80:  10600 < q < 10800 4.5 Hz
M = 0.82:  
M = 0.84: < 17500 (divergence!) 
"""

"""
AOA = 3 deg 

M = 0.74: 9020
M = 0.76: 9450  4.16 Hz 
M = 0.78: 10900 4.33 Hz 
M = 0.80:  10600 < q < 10800 4.5 Hz
M = 0.82:  10800 4.33 Hz  
M = 0.84: < 17500 (divergence!) 
"""

# 2D medium, fine grid + SA options 
aoa_1 = [0., 1., 2., 3., 4., 4.5, 5., 6.]
q_flutter_1 = np.array([9650., 10075., 10550., 10100., 7250., 7300., 6600., 2900. ])/47.5
q_flutter_2 = np.array([9450., 10375., 10900., 12875., 9500., 7900., 5250., 3300. ])/47.5
q_f_error_1 = np.array([100., 100., 100., 100., 100., 100., 150., 150.])/47.5
q_f_error_2 = np.array([100., 100., 100., 100., 100., 100., 150., 150.])/47.5

aoa_SAopt = [0., 1., 2., 3., 4., 5., 6.]
q_flutter_SAopt = np.array([9125., 9750., 10375., 12400., 5500., 0., 0.])/47.883
f_flutter_SAopt = np.array([3.85, 3.85, 3.90, 4.0, 4.7, 5.2, 5.2])
q_f_error_SAopt = np.array([100., 100., 100., 100., 100., 0., 0.])/47.883

f_flutter_1 = np.array([3.90, 3.95, 4.1, 4.3, 4.5, 4.65, 4.80, 5.20])
f_error_1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

f_flutter_2 = np.array([3.90, 3.95, 4.2, 4.3, 4.7, 4.75, 5.0, 5.15])
f_error_2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Bret's results 
aoa_FUN3D = np.array([0., 1., 2., 3., 4., 5.])
q_flutter_FUN3D = np.array([187., 213.,290., 245.,145., 26.])


#f_flutter_SAopt = np.array([4.0, 4.75, 5.2, 5.2])
f_error_SAopt = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

aoa_sa = [0., 1., 2., 3., 4., 5.]
aoa_sst = [0., 1., 2., 3., 4., 5., 6.]
q_flutter_sst = np.array([259., 406., 460., 405., 284., 150., 14.])
q_flutter_sa = np.array([297., 282., 388., 290., 145., 2.])

mach = np.array([0.74, 0.76, 0.78, 0.80, 0.82, 0.84])
q_flutter_machsweep = np.array([9020., 9450., 10900., 10700., 10800., 17500. ])/47.5
f_flutter_mach = np.array([4.10, 4.16, 4.33, 4.5, 4.33, 4.20])
q_f_error_machsweep = np.array([100., 100., 100., 100., 100., 100.])/47.5
f_error_machsweep = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

plt.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    # Axes & ticks
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,

    # Grid
    "axes.grid": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,

    # Lines & markers
    "lines.linewidth": 1.2,

    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# --- Colors: accessible, journal-safe (works in B&W too) ---
COLOR_MEDIUM = "#2166ac"   # Blue
COLOR_FINE   = "#d6604d"   # Red-orange

fig, ax = plt.subplots(figsize=(6.0, 3.6))   # ~1-column width for most journals

ax.errorbar(
    aoa_1, q_flutter_1, yerr=q_f_error_1,
    color=COLOR_MEDIUM, marker="o", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=2, label="Medium Grid SA",
    zorder=3
)
ax.errorbar(
    aoa_1, q_flutter_2, yerr=q_f_error_2,
    color=COLOR_FINE, marker="s", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=2, linestyle="-", label="Fine Grid SA",
    zorder=3
)

ax.errorbar(
    aoa_SAopt, q_flutter_SAopt, yerr=q_f_error_SAopt,
    color="green", marker="s", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=2, linestyle="-", label="Fine Grid, SA neg + compr",
    zorder=3
)

plt.plot(aoa_FUN3D,q_flutter_FUN3D,'k-',marker="D",linewidth=2,label="NASA FOM FUN3D")

ax.set_xlabel(r'Angle of Attack, $\alpha$ (deg)')
ax.set_ylabel(r'Flutter Dynamic Pressure, $q_\mathrm{f}$ (psf)')
###ax.set_title(r'BSCW, $M = 0.80$, SU2 Response', pad=6)

ax.set_xlim(-0.5, 6.5)
ax.set_ylim(0, 320)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))

legend = ax.legend(
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
    handlelength=2.0,
)

plt.tight_layout(pad=0.5)
plt.savefig("flutter_pressure_pub.pdf")   # vector PDF for submission
plt.savefig("flutter_pressure_pub.png", dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(6.0, 3.6))   # ~1-column width for most journals

ax.errorbar(
    aoa_1, f_flutter_1, yerr=f_error_1,
    color=COLOR_MEDIUM, marker="o", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=2, label="Medium Grid SA",
    zorder=3
)
ax.errorbar(
    aoa_1, f_flutter_2, yerr=f_error_2,
    color=COLOR_FINE, marker="s", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=2, linestyle="-", label="Fine Grid SA",
    zorder=3
)

ax.errorbar(
    aoa_SAopt, f_flutter_SAopt, yerr=f_error_SAopt,
    color="green", marker="s", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=2, linestyle="-", label="Fine Grid SA neg + compr",
    zorder=3
)

ax.set_xlabel(r'Angle of Attack, $\alpha$ (deg)')
ax.set_ylabel(r'Flutter Frequency, $f_\mathrm{f}$ (Hz)')
#ax.set_title(r'BSCW, $M = 0.80$, SU2 Response', pad=6)

ax.set_xlim(-0.5, 6.5)
ax.set_ylim(3.00, 6.00)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

legend = ax.legend(
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
    handlelength=2.0,
)

plt.tight_layout(pad=0.5)
plt.savefig("flutter_freq.pdf")   # vector PDF for submission
plt.savefig("flutter_freq.png", dpi=300)
plt.show()



fig, ax = plt.subplots(figsize=(6, 3.6))   # ~1-column width for most journals

ax.errorbar(
    mach, q_flutter_machsweep,yerr=q_f_error_machsweep,
    color=COLOR_MEDIUM, marker="o", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=1.2, label="Coarse Grid",
    zorder=3
)

# plt.plot(mach,q_flutter_machsweep)

ax.set_xlabel(r'Mach Number, $M (\,)$')
ax.set_ylabel(r'Flutter Dynamic Pressure, $q_\mathrm{f}$ (psf)')

# ax.set_xlim(-0.5, 6.5)
# ax.set_ylim(3.00, 6.00)

ax.set_xlim(0.73, 0.85)
ax.set_ylim(100, 410)

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.0025))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))

legend = ax.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
    handlelength=2.0,
)

plt.tight_layout(pad=0.5)
plt.savefig("flutter_freq_machsweep.pdf")   # vector PDF for submission
plt.savefig("flutter_freq_machsweep.png", dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(6.0, 3.6))   # ~1-column width for most journals

ax.errorbar(
    mach, f_flutter_mach, yerr=f_error_machsweep,
    color=COLOR_MEDIUM, marker="o", markersize=4,
    capsize=3, capthick=0.8, elinewidth=0.8,
    linewidth=1.2, label="Coarse Grid",
    zorder=3
)

ax.set_xlabel(r'Mach Number, $M (\,)$')
ax.set_ylabel(r'Flutter Frequency, $f_\mathrm{f}$ (Hz)')
#ax.set_title(r'BSCW, $M = 0.80$, SU2 Response', pad=6)


ax.set_xlim(0.73, 0.85)
ax.set_ylim(3.00, 6.00)

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.0025))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

legend = ax.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
    handlelength=2.0,
)

plt.tight_layout(pad=0.5)
plt.savefig("flutter_freq_machsweep.pdf")   # vector PDF for submission
plt.savefig("flutter_freq_machsweep.png", dpi=300)
plt.show()