import matplotlib.pyplot as plt
import numpy as np

# SA plain 
# 0 deg: >9500
# 1 deg: <10200
# 2 deg: 10500
# 3 deg: < 10500
# 4 deg: < 6000
# 4.5 deg: < 8000 > 7000???
# 5 deg: < 6700???
# 6 deg: <1000, 5 H

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

# Data
aoa_1 = [0., 1., 2., 3., 4., 4.5, 5., 6.]
q_flutter_1 = np.array([9400., 10000., 10500., 10200., 6000., 6000., 5000., 1000. ])/47.5
q_flutter_2 = np.array([9250., 10500., 11230., 10700., 9500., 7900., 0., 3300. ])/47.5
q_f_error_1 = np.array([100., 250., 350., 500., 2000., 2000., 2000., 200.])/47.5
q_f_error_2 = np.array([100., 250., 100., 200., 200., 200., 0., 400.])/47.5


aoa_sa = [0., 1., 2., 3., 4., 5.]
aoa_sst = [0., 1., 2., 3., 4., 5., 6.]
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
plt.errorbar(aoa_1,q_flutter_1,q_f_error_1, linewidth=2,label="Medium Grid")
plt.errorbar(aoa_1,q_flutter_2,q_f_error_2, linewidth=2,  label="Fine Grid")

# plt.plot(aoa_1,q_flutter_2,label="fine grid")
# plt.plot(aoa_sst,q_flutter_sst,label="SST step")


#plt.plot(aoa_1, q_flutter1, 'o-', color='#1f77b4', linewidth=2, 
#        markersize=7, label='SA', clip_on=False)
#plt.plot(aoa_sst, q_flutter_sst, 's--', color='#d62728', linewidth=2, 
#        markersize=7, label='SST', clip_on=False)

# Labels with LaTeX formatting
plt.xlabel(r'Angle of Attack, $\alpha$ [$^\circ$]')
plt.ylabel(r'Flutter Dynamic Pressure, $q_{flutter}$ [psf]')
plt.title('BSCW M=0.80 with SU2 response')

# Formatting
plt.grid(True)
plt.legend(loc='best', frameon=True, shadow=False)
plt.xlim(-0.5, 6.5)
plt.ylim(0, 280.) # Providing space at the top


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
