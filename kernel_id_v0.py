import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


Mach = 0.8      # freestream Mach number
AOA = 5         # wind off angle-of-attack [deg]
v_inf = 131.41  # freestream velocity from CFD simulation [m/s]
rho_inf = 0.832 # freestream density from CFD simulation [kg/m^3]
q_inf = 0.5*rho_inf*v_inf**2 
delta_t = 0.0002 # transient time step [seconds]
chord = 0.41

gen_static_forces = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/gen_static_force_AOA%s_q150psf_SST_med_test.txt" % AOA) # generalised static force normalised by q_inf

print(gen_static_forces)

gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_med_test.txt"%AOA) # INPUTS: generalized structural displacements
heave_input_int = 1e0*gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
pitch_input = 1e0*gen_disp[1,:]*29.85872 /180*2*np.pi # pitch input if you want to convert to deg
ntsteps = len(pitch_input)
time_vector = np.linspace(0.,delta_t*ntsteps,ntsteps)
tau_vector = time_vector*2*v_inf/chord

heave_input = 1.0*heave_input_int #np.gradient(heave_input_int,time_vector)

print("NTSTEPS = ",ntsteps)

gen_force = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_med_test.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf

M_aa = np.array([[ 1.00000000e+00, -8.98073212e-10],
                [-8.98073227e-10,  9.99999999e-01]])

K_aa = np.array([[4.37742677e+02, 2.01950251e-05],
                [1.18480329e-06, 1.06733307e+03]])

C_aa_critical = 2*np.sqrt(np.abs(M_aa*K_aa)) 

# gen_stiffness_matrix = np.array([[ 1.00000000e+00, -8.98073212e-10],
#                                 [-8.98073227e-10,  9.99999999e-01]])

# gen_stiffness_matrix = np.array([[4.37742677e+02, 2.01950251e-05],
#                                  [1.18480329e-06, 1.06733307e+03]])

print(gen_force.shape)

fig, ax = plt.subplots(4, 1, figsize=(10, 10))

plt.subplot(4, 1, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector,gen_force[0,:],label="Gen Force 1")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(tau_vector,gen_force[1,:],label="Gen Force 2")
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

m = 180 

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

print(theta_cl)

print(cltest_check.shape)

delta_cl = + gen_force[0,:] - cltest_check 
delta_cm = + gen_force[1,:] - cmtest_check 

A_pitch2 = A_pitch*A_pitch
A_heave2 = A_heave*A_heave
A_pitch_heave = A_pitch*A_heave

A2 = np.hstack( (np.hstack((A_pitch2,A_heave2)),A_pitch_heave) )

uu, s, vh = linalg.svd(A2,full_matrices=False)    

theta_cl2 = vh.T@(np.diag(1/s)@(uu.T@delta_cl))
theta_cm2 = vh.T@(np.diag(1/s)@(uu.T@delta_cm))

cltest_check2 = cltest_check + A2@theta_cl2
cmtest_check2 = cmtest_check + A2@theta_cm2

fig, ax = plt.subplots(4, 1, figsize=(10, 10))

plt.subplot(4, 1, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector,gen_force[0,:],label="Cl CFD")
plt.plot(tau_vector,cltest_check,label="Cl kernel 1")
plt.plot(tau_vector,cltest_check2,label="Cl kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(tau_vector,gen_force[1,:],label="Cm CFD")
plt.plot(tau_vector,cmtest_check,label="Cm kernel 1")
plt.plot(tau_vector,cmtest_check2,label="Cm kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)

plt.bar(np.arange(m*2),theta_cl)
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)

plt.bar(np.arange(m*2),theta_cm)
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 2DOF SS system 

ndof = 2*m + 4 
ndof_aero = 2*m 

AA = np.zeros((ndof,ndof))
BB = np.zeros((ndof,ndof))

xi = 0.020

n_dof = 2 # physical DOFs

C_aa = xi * C_aa_critical 

Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

nt_int = 12000 # numerical integration steps 

DT = delta_t*1.0 
dtau = delta_t*(2*v_inf)/chord

tau_int = np.linspace(0,nt_int*dtau,nt_int)
time_int = np.linspace(0,nt_int*DT,nt_int)
    
# aerodynamic forces contribution by (n+1) aoa_eff

q = 0.000751

beta = 0.5 

# Aphys[n_dof,0] += q*theta_cl[0]*beta # heave input coeff 1
# Aphys[n_dof+1,0] += q*theta_cm[0]*beta # heave input coeff 1
# Aphys[n_dof,1] += q*theta_cl[m]*beta # pitch input coeff 1
# Aphys[n_dof+1,1] += q*theta_cm[m]*beta # pitch input coeff 1

# discrete time 

Aphys_D = Bphys + DT*(1.-beta)*Aphys
Bphys_D = Bphys - DT*beta*Aphys 

print(Bphys_D)

# top left partition of states materix

AA_D = linalg.inv(Bphys_D)@Aphys_D

# top right partition of states matrix 

BZ = np.zeros((2*n_dof,ndof_aero))

# 2nd order accurate
BZ[n_dof,:] = q * theta_cl # q* ( (1.-beta )*(theta_cl) + beta*np.hstack([theta_cl,0.])[1:])
BZ[n_dof+1,:] = q * theta_cm #q*( (1. - beta)*theta_cm + beta*np.hstack([theta_cm,0.])[1:])

# 1st order accurate
# BZ[n_dof,:] = q*area*chord* (kernels_cm)
# BZ[n_dof+1,:] = q*area* (    kernels_cl)

BZ_D = linalg.inv(Bphys_D)@BZ

# assembly of state matrix 

AA[0:2*n_dof,0:2*n_dof] = AA_D 
AA[0:2*n_dof,2*n_dof:] = DT*BZ_D

# aoa_eff value update 

AA[2*n_dof,0] = 1. # heave at time "n" 
AA[2*n_dof+m,1] = 1. # pitch at time "n" 

AA[2*n_dof:,2*n_dof:] = np.diag(np.ones(ndof_aero-1),-1)
AA[2*n_dof+m,2*n_dof+m-1] = 0. 

# time integration 

qq = np.zeros((ndof,nt_int))

qping = 1.0 # heave 
qq[2,0] = qping # PING
qq[3,0] = qping # PING

fNL = np.zeros(ndof)


for it in range(1,nt_int):
    # if flutter_study == 'NL':       
    #     fNL[2] = q*area*chord*(kernels_cm_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
    #     fNL[3] = q*area*(kernels_cl_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
    #     qq[:,it] = AA@(qq[:,it-1] + DT*fNL) 
    # if flutter_study == 'lin':
        qq[:,it] = AA@qq[:,it-1] 
    #qq[:,it] = AA@qq[:,it-1] 

fig, ax = plt.subplots(2,1)
#fig.suptitle(' Flutter Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))

ax[0].plot(time_int,qq[1,:],'k-',linewidth=2,label=r"$\theta$")
ax[0].set_xlabel(r'$t \,\, [s]$')
ax[0].set_ylabel(r'$\theta \,\, [RAD]$')

ax[1].plot(time_int,qq[0,:],'k-',linewidth=2,label=r"$z$")
ax[1].set_xlabel(r'$t \,\, [s]$')
ax[1].set_ylabel(r'$z \,\, [m]$')
#ax.set_xlim([-10,5])
#ax.set_ylim([0,120])
#ax.legend()
plt.show()

# ####
# Xpitch = np.fft.fft(qq[0,:])
# Xplunge = np.fft.fft(qq[1,:])
# N = len(Xpitch)
# n = np.arange(N)
# T = N*dtau*chord/(2*v_inf)
# freq = n/T 

# plt.figure(figsize = (12, 6))
# plt.subplot()

# plt.stem(freq, Xpitch.real, 'b', \
#         markerfmt="*-", basefmt="-b",label="pitch real")
# plt.stem(freq, Xpitch.imag, 'b', \
#         markerfmt="*-", basefmt="-b",label="pitch imag")
# plt.stem(freq, Xplunge.real, 'k', \
#          markerfmt="ko", basefmt="-b",label="plunge real")
# plt.stem(freq, Xplunge.imag, 'k', \
#          markerfmt="go", basefmt="-b",label="plunge imag")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('FFT Amplitude |X(freq)|')
# plt.xlim(0.0, 10.0)

# plt.legend()
# plt.show()

