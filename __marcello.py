import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg



Mach = 0.8      # freestream Mach number
AOA = 5         # wind off angle-of-attack [deg]
v_inf = 131.41  # freestream velocity from CFD simulation [m/s]
rho_inf = 0.832 # freestream density from CFD simulation [kg/m^3]
q_inf = 0.5*rho_inf*v_inf**2 
delta_t = 0.0002 # transient time step [seconds]

gen_static_forces = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/gen_static_force_AOA%s_q150psf_SST_med_test.txt" % AOA) # generalised static force normalised by q_inf

gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_med_test.txt"%AOA) # INPUTS: generalized structural displacements
heave_input = gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
pitch_input = gen_disp[1,:]*29.85872 # pitch input if you want to convert to deg

ntsteps = len(heave_input)
print("NTSTEPS = ",ntsteps)

chord = 0.41
time_vector = np.linspace(0.,delta_t*ntsteps,ntsteps)
tau_vector = time_vector*2*v_inf/chord

gen_force = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_med_test.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf


gen_stiffness_matrix = np.array([[ 1.00000000e+00, -8.98073212e-10],
                                [-8.98073227e-10,  9.99999999e-01]])

gen_stiffness_matrix = np.array([[4.37742677e+02, 2.01950251e-05],
                                 [1.18480329e-06, 1.06733307e+03]])

print(gen_force.shape)


fig, ax = plt.subplots(4, 1, figsize=(10, 10))

plt.subplot(4, 1, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector,gen_force[0,:])
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(tau_vector,gen_force[1,:])
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)

plt.plot(time_vector,heave_input,label="heave input")
plt.xlabel(r'$t\,[s]$')
plt.ylabel(r'$h\,[m]$')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)

plt.plot(time_vector,pitch_input,label="pitch input")
plt.xlabel(r'$t\,[s]$')
plt.ylabel(r'$\theta [DEG]$')
plt.legend()
plt.grid(True)

# plt.subplot(1, 2, 2)

# plt.title("LAMBDA = "+str(eigvals[1]))
# plt.plot(R[:,1], 'o-',linewidth=2,label="ONSET EXP")
# plt.xlabel(r'$V\,[m/s]$')
# plt.ylabel(r'$\delta/s\, [\,]$')
# plt.grid(True)

plt.tight_layout()
plt.show()

m = 220 

A_pitch = np.zeros((ntsteps,m))
A_heave = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn]
    A_heave[icolumn:,icolumn] = heave_input[:ntsteps-icolumn]

A = np.hstack((A_pitch,A_heave))

uu, s, vh = linalg.svd(A,full_matrices=False)    

theta_cl = vh.T@(np.diag(1/s)@(uu.T@gen_force[0,:]))
theta_cm = vh.T@(np.diag(1/s)@(uu.T@gen_force[1,:]))

cltest_check = A@theta_cl
cmtest_check = A@theta_cm

print(theta_cl.shape)

print(cltest_check.shape)


delta_cl = - cltest_check + gen_force[0,:]
delta_cm = - cmtest_check + gen_force[1,:]

A_pitch2 = A_pitch*A_pitch
A_heave2 = A_heave*A_heave
A_pitch_heave = A_pitch*A_heave

A2 = np.hstack( (np.hstack((A_pitch2,A_heave2)),A_pitch_heave) )

uu, s, vh = linalg.svd(A2,full_matrices=False)    

theta_cl2 = vh.T@(np.diag(1/s)@(uu.T@delta_cl))
theta_cm2 = vh.T@(np.diag(1/s)@(uu.T@delta_cm))

cltest_check2 = cltest_check + A2@theta_cl2
cmtest_check2 = cmtest_check + A2@theta_cm2

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

plt.subplot(2, 1, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector,gen_force[0,:],label="CFD")
plt.plot(tau_vector,cltest_check,label="kernel 1")
plt.plot(tau_vector,cltest_check2,label="kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)

plt.plot(tau_vector,gen_force[1,:],label="CFD")
plt.plot(tau_vector,cmtest_check,label="kernel 1")
plt.plot(tau_vector,cmtest_check2,label="kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





