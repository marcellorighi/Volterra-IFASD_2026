import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize_scalar
from scipy.linalg import block_diag


Mach = 0.8      # freestream Mach number
AOA = 5    # wind off angle-of-attack [deg]

q =  1500.

v_inf = 277.  # freestream velocity from CFD simulation [m/s]
rho_inf = 0.832 # freestream density from CFD simulation [kg/m^3]
q_inf = 0.5*rho_inf*v_inf**2 
delta_t = 0.00075 # transient time step [seconds]
chord = 0.41

dtau = delta_t * v_inf /(2 * chord)
resample_steps = 1 #4 #4 #10
m = 180 #160 #80 #80

ntsteps     = 10000
time_vector = np.linspace(0.,ntsteps*delta_t, ntsteps)

# multisine 

def generate_schroeder_multisine(freqs, dt, duration, amp=1.0):
    t = np.arange(0, duration, dt)
    N = len(freqs)
    # Schroeder Phase Formula: phi_k = -k*(k-1)*pi / N
    phases = [-k * (k - 1) * np.pi / N for k in range(  1, N + 1)]
    
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += amp * np.sin(2 * np.pi * f * t + phases[i])
    
    # Normalize to keep within original amplitude constraints
    # signal = signal / np.max(np.abs(signal)) * amp
    return t, signal

#target_freqs = [2.9, 5.1, 8.0, 11.5] #, 15.9] 
target_freqs_pitching = 11. / (2.*np.pi) * np.array([1.0, 1.47, 2.31, 3.57]) 
target_freqs_plunging = 12. / (2.*np.pi) * np.array([1.0, 1.4142, 2.41, 3.77])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, delta_t, ntsteps*delta_t, amp = 0.50 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, delta_t, ntsteps*delta_t, amp = 0.10)

zdot_input = - np.gradient(plunge_input,time_vector)/v_inf
thetadot_input = np.gradient(pitch_input, time_vector)/v_inf

# 3. Calculate FFT
fft_freqs = np.fft.fftfreq(ntsteps, d = delta_t)

# 4. Filter for positive frequencies only (0 to Nyquist)
positive_mask = fft_freqs >= 0
freqs = fft_freqs[positive_mask]
# Multiply by 2/n to get the actual physical amplitude
fft_values = np.fft.fft(pitch_input)
Pitch_input = 2.0/ntsteps * np.abs(fft_values[positive_mask])

fft_values = np.fft.fft(zdot_input)
Zdot_input = 2.0/ntsteps * np.abs(fft_values[positive_mask])

fft_values = np.fft.fft(thetadot_input)
Thetadot_input = 2.0/ntsteps * np.abs(fft_values[positive_mask])

# 5. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
plt.rcParams.update({"font.family": "serif"})

# Time Domain
ax1.plot(time_vector, np.rad2deg(pitch_input), color='#1f77b4', lw=1,label="PITCH")
ax1.plot(time_vector, np.rad2deg(zdot_input), color='green', lw=1,label="ZDOT")
ax1.plot(time_vector, np.rad2deg(thetadot_input), 'k-', lw=1,label="THETADOT")
ax1.set_title('Time Domain (Experimental Signal)')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')
ax1.legend()

# Frequency Domain
ax2.stem(freqs, Pitch_input, basefmt=" ", linefmt='r-', markerfmt='ro')
ax2.stem(freqs, Zdot_input, basefmt=" ", linefmt='r-', markerfmt='go')
ax2.stem(freqs, Thetadot_input, basefmt=" ", linefmt='b-', markerfmt='b*')
ax2.set_title('Frequency Domain (FFT Analysis)')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Strength')
ax2.set_xlim(0, 50) # Looking at the range of interest
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



