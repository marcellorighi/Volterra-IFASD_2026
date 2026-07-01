import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize_scalar
from scipy.linalg import block_diag


Mach = 0.8      # freestream Mach number
AOA = 5    # wind off angle-of-attack [deg]

q =  1500.

# FINE 3 deg: 10900, 4 deg: 9000. 5 deg: 3150
# MEDIUM  5 deg: 1500. #4 deg: 7800. #3 deg: 10600. #2 deg: 9350. # 1 deg: 9350. #0 deg: 9400. 

# MEDIUM: 0DEG 8600 1DEG 8400 2deg 8900  3deg 10300 

v_inf = 131.41  # freestream velocity from CFD simulation [m/s]
rho_inf = 0.832 # freestream density from CFD simulation [kg/m^3]
q_inf = 0.5*rho_inf*v_inf**2 
delta_t = 0.0002 # transient time step [seconds]
chord = 0.41

dtau = delta_t * v_inf /(2 * chord)
resample_steps = 1 #4 #4 #10
m = 180 #160 #80 #80

print("Model memory: ", resample_steps * m * dtau, " (reduced time units)" )

gen_static_forces = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/gen_static_force_AOA%s_q150psf_SST_med_train.txt" % AOA) # generalised static force normalised by q_inf

# print(gen_static_forces)


# medium mesh 

gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_med_train.txt"%AOA) # INPUTS: generalized structural displacements
gen_force = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_med_train.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf

# fine mesh 

#gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_fine_train.txt"%AOA) # INPUTS: generalized structural displacements
#gen_force = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_fine_train.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf


pitch_input           = 1e0*gen_disp[1,:]*29.85872 /180*np.pi # pitch input if you want to convert to deg
pitch_input_resampled = 1e0*gen_disp[1,::resample_steps]*29.85872 /180*np.pi # pitch input if you want to convert to deg

ntsteps            = len(pitch_input)

time_vector         = np.linspace(0.,ntsteps*delta_t, ntsteps)

# multisine 

def generate_schroeder_multisine(freqs, dt, duration, amp=1.0):
    t = np.arange(0, duration, dt)
    N = len(freqs)
    # Schroeder Phase Formula: phi_k = -k*(k-1)*pi / N
    phases = [-k * (k - 1) * np.pi / N for k in range(1, N + 1)]
    
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += amp * np.cos(2 * np.pi * f * t + phases[i])
    
    # Normalize to keep within original amplitude constraints
    signal = signal / np.max(np.abs(signal)) * amp
    return t, signal

#target_freqs = [2.9, 5.1, 8.0, 11.5] #, 15.9] 
target_freqs = [1.5, 3.21, 5.1, 10.05] #, 15.9] 
t, aoa1 = generate_schroeder_multisine(target_freqs, delta_t, ntsteps*delta_t, amp = 3 * np.pi/180.)

# 3. Calculate FFT
signal_1 = np.gradient(gen_disp[0,:],time_vector)/v_inf *0.106654261 
signal_2 = gen_disp[1,:]*29.85872 /180*np.pi
fft_freqs = np.fft.fftfreq(ntsteps, d = delta_t)

# 4. Filter for positive frequencies only (0 to Nyquist)
positive_mask = fft_freqs >= 0
freqs = fft_freqs[positive_mask]
# Multiply by 2/n to get the actual physical amplitude
fft_values = np.fft.fft(signal_1)
amplitude_1 = 2.0/ntsteps * np.abs(fft_values[positive_mask])

fft_values = np.fft.fft(signal_2)
amplitude_2 = 2.0/ntsteps * np.abs(fft_values[positive_mask])

# multisine FFT
fft_values = np.fft.fft(aoa1)
aoa1_amplitude = 2.0/ntsteps * np.abs(fft_values[positive_mask])


# 5. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
plt.rcParams.update({"font.family": "serif"})

# Time Domain
ax1.plot(time_vector, signal_1, color='#1f77b4', lw=1)
ax1.plot(time_vector, signal_2, color='green', lw=1)
ax1.plot(time_vector, aoa1, 'k-', lw=1)
ax1.set_title('Time Domain (Experimental Signal)')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')

# Frequency Domain
ax2.stem(freqs, amplitude_1, basefmt=" ", linefmt='r-', markerfmt='ro')
ax2.stem(freqs, amplitude_2, basefmt=" ", linefmt='r-', markerfmt='go')
ax2.stem(freqs, aoa1_amplitude, basefmt=" ", linefmt='b-', markerfmt='b*')
ax2.set_title('Frequency Domain (FFT Analysis)')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Strength')
ax2.set_xlim(0, 50) # Looking at the range of interest
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

