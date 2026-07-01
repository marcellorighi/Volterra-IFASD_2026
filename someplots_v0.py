import matplotlib.pyplot as plt
import numpy as np

# Data
aoa_FUN3D = np.array([0., 1., 2., 3., 3.5, 4., 5., 6.])
qflutter_FUN3D = np.array([165.72, 161.93, 173.77, 206.91, 194.60, 142.52, 119.32, 25.09])
fflutter_FUN3D = np.array([4.14, 4.04, 4.11, 4.28, 4.49, 4.72, 5.06, 4.99])

aoa1 = [0., 1., 2., 3., 4., 5.]
# q_flutter1 = np.array([9400., 9350., 9350., 10600., 7800., 1500.])/47.5
# q_flutter1 = np.array([8550., 8100., 8700., 10700., 8800., 4250.])/47.883
q_flutter1 = np.array([8050., 7950., 8550., 10250., 9375., 4250.])/47.883
f_flutter1 = np.array([24.5/6, 4.06, 4.09, 4.14, 4.50, 5.])

aoa2 = [3., 4., 5.]
# q_flutter2 = np.array([10900., 9000., 3150.])/47.5
q_flutter2 = np.array([10070., 9200., 5300.])/47.883
f_flutter2 = np.array([4.14, 4.5, 4.8])

# WT data 2026
x = 5.
y1 = 50.
y2 = 180. 
y_center = (y1 + y2) / 2
lower_error = y_center - y1
upper_error = y2 - y_center
y_err = [[lower_error], [upper_error]]
# WT data 1990s
m_wt = np.array([0., 1.28, 5.30, 5.40, 5.50])
q_wt = np.array([171., 165., 106., 123., 94.])
f_wt = np.array([4.06, 4.03, 4.83, 4.85, 4.98])

# Professional styling configuration
plt.rcParams.update({
    "font.family": "serif",       # Serif fonts are standard for technical publications
    "font.size": 11,
    "axes.labelsize": 12,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

fig, ax = plt.subplots(1,2,figsize=(12, 3.6))


plt.subplot(1, 2, 1)

# Plotting with distinct markers and colors
plt.errorbar(x, y_center, yerr=y_err, fmt='none', ecolor='black', elinewidth=4, capsize=6, label="NASA TDT exp. 2026")
plt.plot(m_wt, q_wt, marker='D', ls='none', ms=8, color='black', 
         markeredgecolor='black', label='NASA TDT exp. 1992')

plt.plot(aoa_FUN3D, qflutter_FUN3D,'ko-', label = "NASA FOM FUN3D")
plt.plot(aoa1, q_flutter1, 'o-', color='#1f77b4', linewidth=2, 
        markersize=7, label='present Medium Grid', clip_on=False)
plt.plot(aoa2, q_flutter2, 's-', color='#d62728', linewidth=2, 
        markersize=7, label='present Fine Grid', clip_on=False)

# Labels with LaTeX formatting
plt.xlabel(r'Angle of Attack, $\alpha$ [$^\circ$]')
plt.ylabel(r'Flutter Dynamic Pressure, $q_{flutter}$ [psf]')
# plt.title('Case 1 (CFD data from Michael Candon)')

# Formatting
plt.grid(True)
plt.legend(loc='best', frameon=True, shadow=False)
plt.xlim(-0.5, 6.5)
plt.ylim(0, 12000/47.5) # Providing space at the top


plt.subplot(1, 2, 2)

plt.plot(m_wt, f_wt, marker='D', ls='none', ms=8, color='black', 
         markeredgecolor='black', label='NASA TDT exp. 1992')

plt.plot(aoa_FUN3D, fflutter_FUN3D, 'ko-', label = "NASA FUN3D")

plt.plot(aoa1, f_flutter1, 'o-', color='#1f77b4', linewidth=2, 
        markersize=7, label='present Medium Grid', clip_on=False)

plt.plot(aoa2, f_flutter2, 'o-',  color='#d62728', linewidth=2, 
        markersize=7, label='present Fine Grid', clip_on=False)

# Labels with LaTeX formatting
plt.xlabel(r'Angle of Attack, $\alpha$ [$^\circ$]')
plt.ylabel(r'FlutterFrequency, $f_{flutter}$ [Hz]')
# plt.title('Case 1 (CFD data from Michael Candon)')

# Formatting
plt.grid(True)
plt.legend(loc='best', frameon=True, shadow=False)
plt.xlim(-0.5, 6.5)
plt.ylim(3.0, 6.0) # Providing space at the top

# Using tight_layout to ensure labels aren't cut off
plt.tight_layout()
plt.show() 

# Save as a high-resolution PNG for documents
#plt.savefig('flutter_plot_bscw_Michael.png', dpi=300)
