import matplotlib.pyplot as plt
import numpy as np

# Data
aoa_sa = [0., 1., 2., 3., 4., 5.]
aoa_sst = [0., 1., 2., 3., 4., 5., 6.]
# q_flutter1 = np.array([9400., 9350., 9350., 10600., 7800., 1500.])/47.5
q_flutter_sst = np.array([259., 406., 460., 405., 284., 150., 14.])
q_flutter_sa = np.array([297., 282., 388., 290., 145., 2.])
# f_flutter1 = np.array([4.0, 3.97, 4.05, 4.11, 4.50, 5.])

# Professional styling configuration
plt.rcParams.update({
    "font.family": "serif",       # Serif fonts are standard for technical publications
    "font.size": 11,
    "axes.labelsize": 12,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

fig, ax = plt.subplots(figsize=(10, 5))


#plt.subplot(1, 2, 1)

# Plotting with distinct markers and colors
plt.plot(aoa_sa, q_flutter_sa, 'o-', color='#1f77b4', linewidth=2, 
        markersize=7, label='SA', clip_on=False)
plt.plot(aoa_sst, q_flutter_sst, 's--', color='#d62728', linewidth=2, 
        markersize=7, label='SST', clip_on=False)

# Labels with LaTeX formatting
plt.xlabel(r'Angle of Attack, $\alpha$ [$^\circ$]')
plt.ylabel(r'Flutter Dynamic Pressure, $q_{flutter}$ [psf]')
plt.title('BSCW M=0.80 with SU2 step response')

# Formatting
plt.grid(True)
plt.legend(loc='best', frameon=True, shadow=False)
plt.xlim(-0.5, 6.5)
plt.ylim(0, 480.) # Providing space at the top


# plt.subplot(1, 2, 2)
# plt.plot(aoa1, f_flutter1, 'o-', color='#1f77b4', linewidth=2, 
#         markersize=7, label='Medium Grid', clip_on=False)

# # Labels with LaTeX formatting
# plt.xlabel(r'Angle of Attack, $\alpha$ [$^\circ$]')
# plt.ylabel(r'FlutterFrequency, $f_{flutter}$ [Hz]')
# plt.title('BSCW M=0.80 with data from Michael Candon')

# # Formatting
# plt.grid(True)
# plt.legend(loc='best', frameon=True, shadow=False)
# plt.xlim(-0.5, 5.5)
# plt.ylim(3.0, 6.0) # Providing space at the top

# # Using tight_layout to ensure labels aren't cut off
# plt.tight_layout()
plt.show() 

# Save as a high-resolution PNG for documents
#plt.savefig('flutter_plot_bscw_Michael.png', dpi=300)
