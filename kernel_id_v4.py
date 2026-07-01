import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


Mach = 0.8      # freestream Mach number
AOA = 5     # wind off angle-of-attack [deg]
v_inf = 131.41  # freestream velocity from CFD simulation [m/s]
rho_inf = 0.832 # freestream density from CFD simulation [kg/m^3]
q_inf = 0.5*rho_inf*v_inf**2 
delta_t = 0.0002 # transient time step [seconds]
chord = 0.41

dtau = delta_t * v_inf /(2 * chord)
resample_steps = 10 
m = 40

print("Model memory: ", resample_steps * m * dtau, " (reduced time units)" )

gen_static_forces = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/gen_static_force_AOA%s_q150psf_SST_med_train.txt" % AOA) # generalised static force normalised by q_inf

# print(gen_static_forces)

gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_med_train.txt"%AOA) # INPUTS: generalized structural displacements

gen_force = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_med_train.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf

pitch_input           = 1e0*gen_disp[1,:]*29.85872 /180*np.pi # pitch input if you want to convert to deg
pitch_input_resampled = 1e0*gen_disp[1,::resample_steps]*29.85872 /180*np.pi # pitch input if you want to convert to deg

ntsteps            = len(pitch_input)
ntsteps_resampled  = len(pitch_input_resampled)

time_vector           = np.linspace(0.,ntsteps*delta_t, ntsteps)
time_vector_resampled = time_vector[::resample_steps] #np.linspace(0.,resample_steps*delta_t*ntsteps_resampled,ntsteps_resampled)

tau_vector = time_vector*2*v_inf/chord
tau_vector_resampled = time_vector_resampled*2*v_inf/chord

heave_input_int = 1e0*gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
heave_input_grad = np.gradient(heave_input_int,time_vector)
heave_input_grad_resampled = heave_input_grad[::resample_steps]/v_inf

pitch_input_grad = np.gradient(gen_disp[1,:]*29.85872 /180*np.pi,time_vector)
pitch_input_grad_resampled = pitch_input_grad[::resample_steps]/v_inf


cl_output = gen_force[0,:] / 0.106654 / (chord*chord*2)
cl_output_resampled = gen_force[0,::resample_steps] / 0.106654 / (chord*chord*2)

cm_output = gen_force[1,:]/ (29.86*np.pi/180) / (chord*chord*chord*2)
cm_output_resampled = gen_force[1,::resample_steps] / (29.86*np.pi/180) / (chord*chord*chord*2)

print("NTSTEPS = ",ntsteps)
print("NTSTEPS_RESAMPLED = ",ntsteps_resampled)

fig, ax = plt.subplots(5, 1, figsize=(10, 10))

plt.subplot(5, 1, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector_resampled,cm_output_resampled,".",label=r"$C_m$")
plt.plot(tau_vector,cm_output,"-",label=r"$C_m$")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$C_m\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 2)

plt.plot(tau_vector_resampled,cl_output_resampled,".",label=r"$C_l$")
plt.plot(tau_vector,cl_output,"-",label=r"$C_l$")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$C_l\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)

plt.plot(time_vector_resampled,heave_input_grad_resampled*180/np.pi,label="heave input")
plt.xlabel(r'$t\,[s]$')
plt.ylabel(r'$\dot{h}/V \,[RAD]$')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)

plt.plot(time_vector_resampled,pitch_input_resampled*180/np.pi,label="pitch input")
plt.xlabel(r'$t\,[s]$')
plt.ylabel(r'$\theta [DEG]$')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 5)

plt.plot(time_vector_resampled,(pitch_input_resampled + heave_input_grad_resampled)*180/np.pi,label="AoA")
plt.xlabel(r'$t\,[s]$')
plt.ylabel(r'$\alpha [DEG]$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


A_pitch = np.zeros((ntsteps,m))
A_thetadot = np.zeros((ntsteps,m))
A_heave = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn]
    A_thetadot[icolumn:,icolumn] = pitch_input_grad[:ntsteps-icolumn]
    A_heave[icolumn:,icolumn] = heave_input_grad[:ntsteps-icolumn]

A = np.hstack((A_pitch,A_thetadot,A_heave))

print("A size", A.shape)
# uu, s, vh = linalg.svd(A,full_matrices=False)    

# truncated SVD??? 
# threshold = 0.01 * s[0]  # 1% of largest singular value
# s_inv = np.where(s > threshold, 1/s, 0)

# theta_cl = vh.T@(np.diag(1/s)@(uu.T@cl_output))
# theta_cm = vh.T@(np.diag(1/s)@(uu.T@cm_output))


# from sklearn.linear_model import Lasso

# lasso = Lasso(alpha=0.01, max_iter=10000)
# lasso.fit(A, cl_output)
# theta_cl = lasso.coef_


from scipy.optimize import minimize_scalar

def l_curve_analysis(A, y, lambdas=None):
    """
    Find optimal lambda via L-curve corner detection
    """
    if lambdas is None:
        lambdas = np.logspace(-6, 2, 50)  # Scan wide range
    
    residuals = []
    solution_norms = []
    
    for lam in lambdas:
        h = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        residuals.append(np.linalg.norm(A @ h - y))
        solution_norms.append(np.linalg.norm(h))
    
    residuals = np.array(residuals)
    solution_norms = np.array(solution_norms)
    
    # # Plot L-curve
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.loglog(residuals, solution_norms, 'b.-')
    # plt.xlabel('Residual Norm ||Ah - y||')
    # plt.ylabel('Solution Norm ||h||')
    # plt.title('L-Curve')
    # plt.grid(True)
    
    # Find corner using curvature
    # Convert to log scale for curvature calculation
    log_res = np.log(residuals)
    log_sol = np.log(solution_norms)
    
    # Curvature formula for parametric curve
    curvature = np.zeros(len(lambdas) - 2)
    for i in range(1, len(lambdas) - 1):
        dx1 = log_res[i] - log_res[i-1]
        dy1 = log_sol[i] - log_sol[i-1]
        dx2 = log_res[i+1] - log_res[i]
        dy2 = log_sol[i+1] - log_sol[i]
        
        # Curvature approximation
        curvature[i-1] = abs(dx1*dy2 - dx2*dy1) / (dx1**2 + dy1**2)**1.5
    
    # Find maximum curvature (corner)
    corner_idx = np.argmax(curvature) + 1
    optimal_lambda = lambdas[corner_idx]
    
    # plt.plot(residuals[corner_idx], solution_norms[corner_idx], 'ro', 
    #          markersize=10, label=f'Corner: λ={optimal_lambda:.2e}')
    # plt.legend()
    
    # # Show curvature
    # plt.subplot(1, 2, 2)
    # plt.semilogx(lambdas[1:-1], curvature, 'g.-')
    # plt.axvline(optimal_lambda, color='r', linestyle='--', label='Optimal λ')
    # plt.xlabel('λ')
    # plt.ylabel('Curvature')
    # plt.title('L-Curve Curvature')
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    
    return optimal_lambda, lambdas, residuals, solution_norms

# Usage
#optimal_lambda, _, _, _ = l_curve_analysis(A, y)

# ------------
lambda_reg, _, _, _ = l_curve_analysis(A, cl_output)
#lambda_reg = 0.1 # 1.
#print("lambda",lambda_reg)

L = np.diag(np.linspace(0, 1, 3*m)**2)  # Quadratic penalty on later lags
theta_cl = np.linalg.solve(A.T @ A + lambda_reg * L, A.T @ cl_output)

lambda_reg, _, _, _ = l_curve_analysis(A, cm_output)
L = np.diag(np.linspace(0, 1, 3*m)**2)  # Quadratic penalty on later lags
theta_cm = np.linalg.solve(A.T @ A + lambda_reg * L, A.T @ cm_output)

cltest_check = A@theta_cl 
cmtest_check = A@theta_cm 

print(theta_cl.shape)

print(cltest_check.shape)


delta_cl = + cl_output - cltest_check 
delta_cm = + cm_output - cmtest_check 

A_pitch2 = A_pitch*A_pitch
A_thetadot2 = A_thetadot*A_thetadot
A_heave2 = A_heave*A_heave
A_pitch_heave = A_pitch*A_heave
A_thetadot_heave = A_thetadot*A_heave

# A2 = np.hstack( (np.hstack((A_pitch2,A_thetadot2,A_heave2)),(A_pitch_heave,A_thetadot_heave)) )
A2 = np.hstack( (A_pitch2,A_thetadot2,A_heave2,A_pitch_heave,A_thetadot_heave)) 

# uu, s, vh = linalg.svd(A2,full_matrices=False)    
lambda_reg, _, _, _ = l_curve_analysis(A2, cl_output)
L = np.diag(np.linspace(0, 1, 5*m)**2)  # Quadratic penalty on later lags
theta_cl2 = np.linalg.solve(A2.T @ A2 + lambda_reg * L, A2.T @ delta_cl)

lambda_reg, _, _, _ = l_curve_analysis(A2, cm_output)
L = np.diag(np.linspace(0, 1, 5*m)**2)  # Quadratic penalty on later lags
theta_cm2 = np.linalg.solve(A2.T @ A2 + lambda_reg * L, A2.T @ delta_cm)

# theta_cl2 = vh.T@(np.diag(1/s)@(uu.T@delta_cl))
# theta_cm2 = vh.T@(np.diag(1/s)@(uu.T@delta_cm))

cltest_check2 = cltest_check + A2@theta_cl2
cmtest_check2 = cmtest_check + A2@theta_cm2

fig, ax = plt.subplots(3, 2, figsize=(10, 8))

plt.subplot(3, 2, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector,cl_output,label="Cl CFD")
plt.plot(tau_vector,cltest_check,label="Cl kernel 1")
plt.plot(tau_vector,cltest_check2,label="Cl kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)

plt.plot(tau_vector,cm_output,label="Cm CFD")
plt.plot(tau_vector,cmtest_check,label="Cm kernel 1")
plt.plot(tau_vector,cmtest_check2,label="Cm kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)

plt.bar(np.arange(m*3),theta_cl)
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)

plt.bar(np.arange(m*3),theta_cm)
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)

plt.bar(np.arange(m*5),theta_cl2)
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)

plt.bar(np.arange(m*5),theta_cm2)
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# test signal 
gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_med_test.txt"%AOA) # INPUTS: generalized structural displacements
gen_force = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_med_test.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf

# heave_input_int = 1e0*gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
# pitch_input = 1e0*gen_disp[1,::resample_steps]*29.85872 /180*np.pi # pitch input if you want to convert to deg

# heave_input_grad = np.gradient(heave_input_int,time_heave)
# heave_input = heave_input_grad[::resample_steps]/v_inf  #1.0*heave_input_int #np.gradient(heave_input_int,time_vector)

# pitch_input_grad = np.gradient(gen_disp[1,:]*29.85872 /180*np.pi,time_heave)
# thetadot_input = pitch_input_grad[::resample_steps]/v_inf

# heave_input = heave_input_grad[::resample_steps]/v_inf  #1.0*heave_input_int #np.gradient(heave_input_int,time_vector)


pitch_input           = 1e0*gen_disp[1,:]*29.85872 /180*np.pi # pitch input if you want to convert to deg
pitch_input_resampled = 1e0*gen_disp[1,::resample_steps]*29.85872 /180*np.pi # pitch input if you want to convert to deg

ntsteps            = len(pitch_input)
ntsteps_resampled  = len(pitch_input_resampled)

time_vector           = np.linspace(0.,ntsteps*delta_t, ntsteps)
time_vector_resampled = time_vector[::resample_steps] #np.linspace(0.,resample_steps*delta_t*ntsteps_resampled,ntsteps_resampled)

tau_vector = time_vector*2*v_inf/chord
tau_vector_resampled = time_vector_resampled*2*v_inf/chord

heave_input_int = 1e0*gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
heave_input_grad = np.gradient(heave_input_int,time_vector)
heave_input_grad_resampled = heave_input_grad[::resample_steps]/v_inf

pitch_input_grad = np.gradient(gen_disp[1,:]*29.85872 /180*np.pi,time_vector)
pitch_input_grad_resampled = pitch_input_grad[::resample_steps]/v_inf


cl_output = gen_force[0,:] / 0.106654 / (chord*chord*2)
cl_output_resampled = gen_force[0,::resample_steps] / 0.106654 / (chord*chord*2)

cm_output = gen_force[1,:]/ (29.86*np.pi/180) / (chord*chord*chord*2)
cm_output_resampled = gen_force[1,::resample_steps] / (29.86*np.pi/180) / (chord*chord*chord*2)


# cl_output = gen_force[0,::resample_steps]
# cm_output = gen_force[1,::resample_steps]

# cl_output_print = cl_output / 0.106654 / (chord*chord*2)
# cm_output_print = cm_output / (29.86*np.pi/180) / (chord*chord*chord*2)

# ntsteps = len(pitch_input)
# time_vector = np.linspace(0.,10*delta_t*ntsteps,ntsteps)
# tau_vector = time_vector*2*v_inf/chord

# Volterra prediction 
A_pitch = np.zeros((ntsteps,m))
A_heave = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn]
    A_thetadot[icolumn:,icolumn] = pitch_input_grad[:ntsteps-icolumn]
    A_heave[icolumn:,icolumn] = heave_input_grad[:ntsteps-icolumn]

A = np.hstack((A_pitch,A_thetadot,A_heave))

A_pitch2 = A_pitch*A_pitch
A_thetadot2 = A_thetadot*A_thetadot
A_heave2 = A_heave*A_heave
A_pitch_heave = A_pitch*A_heave
A_thetadot_heave = A_thetadot*A_heave

A2 = np.hstack( (A_pitch2,A_thetadot2,A_heave2,A_pitch_heave,A_thetadot_heave)) 

cltest_check = A@theta_cl #/ 0.106654 / (chord*chord*2)
cmtest_check = A@theta_cm #/ (29.86*np.pi/180) / (chord*chord*chord*2)
cltest_check2 = cltest_check + (A2@theta_cl2) #/ 0.106654 / (chord*chord*2)
cmtest_check2 = cmtest_check + (A2@theta_cm2) #/ (29.86*np.pi/180) / (chord*chord*chord*2)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

plt.subplot(2, 1, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector,cl_output,label="Cl CFD")
plt.plot(tau_vector,cltest_check,label="Cl kernel 1")
plt.plot(tau_vector,cltest_check2,label="Cl kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$C_l\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)

plt.plot(tau_vector,cm_output,label="Cm CFD")
plt.plot(tau_vector,cmtest_check,label="Cm kernel 1")
plt.plot(tau_vector,cmtest_check2,label="Cm kernel 2")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$C_m\, [\,]$')
plt.legend()
plt.grid(True)

plt.tight_layout() 
plt.show()



# # 2DOF SS system 

# ndof = 2*m + 4 
# ndof_aero = 2*m 

# AA = np.zeros((ndof,ndof))
# BB = np.zeros((ndof,ndof))

# xi = 0.020

# n_dof = 2 # physical DOFs

# C_aa = xi * C_aa_critical 

# Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
# Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

# nt_int = 12000 # numerical integration steps 

# DT = delta_t*1.0 
# dtau = delta_t*(2*v_inf)/chord

# tau_int = np.linspace(0,nt_int*dtau,nt_int)
# time_int = np.linspace(0,nt_int*DT,nt_int)
    
# # aerodynamic forces contribution by (n+1) aoa_eff

# q = 0.000751

# beta = 0.5 

# # Aphys[n_dof,0] += q*theta_cl[0]*beta # heave input coeff 1
# # Aphys[n_dof+1,0] += q*theta_cm[0]*beta # heave input coeff 1
# # Aphys[n_dof,1] += q*theta_cl[m]*beta # pitch input coeff 1
# # Aphys[n_dof+1,1] += q*theta_cm[m]*beta # pitch input coeff 1

# # discrete time 

# Aphys_D = Bphys + DT*(1.-beta)*Aphys
# Bphys_D = Bphys - DT*beta*Aphys 

# print(Bphys_D)

# # top left partition of states materix

# AA_D = linalg.inv(Bphys_D)@Aphys_D

# # top right partition of states matrix 

# BZ = np.zeros((2*n_dof,ndof_aero))

# # 2nd order accurate
# BZ[n_dof,:] = q * theta_cl # q* ( (1.-beta )*(theta_cl) + beta*np.hstack([theta_cl,0.])[1:])
# BZ[n_dof+1,:] = q * theta_cm #q*( (1. - beta)*theta_cm + beta*np.hstack([theta_cm,0.])[1:])

# # 1st order accurate
# # BZ[n_dof,:] = q*area*chord* (kernels_cm)
# # BZ[n_dof+1,:] = q*area* (    kernels_cl)

# BZ_D = linalg.inv(Bphys_D)@BZ

# # assembly of state matrix 

# AA[0:2*n_dof,0:2*n_dof] = AA_D 
# AA[0:2*n_dof,2*n_dof:] = DT*BZ_D

# # aoa_eff value update 

# AA[2*n_dof,0] = 1. # heave at time "n" 
# AA[2*n_dof+m,1] = 1. # pitch at time "n" 

# AA[2*n_dof:,2*n_dof:] = np.diag(np.ones(ndof_aero-1),-1)
# AA[2*n_dof+m,2*n_dof+m-1] = 0. 

# # time integration 

# qq = np.zeros((ndof,nt_int))

# qping = 1.0 # heave 
# qq[2,0] = qping # PING
# qq[3,0] = qping # PING

# fNL = np.zeros(ndof)


# for it in range(1,nt_int):
#     # if flutter_study == 'NL':       
#     #     fNL[2] = q*area*chord*(kernels_cm_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
#     #     fNL[3] = q*area*(kernels_cl_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
#     #     qq[:,it] = AA@(qq[:,it-1] + DT*fNL) 
#     # if flutter_study == 'lin':
#         qq[:,it] = AA@qq[:,it-1] 
#     #qq[:,it] = AA@qq[:,it-1] 

# fig, ax = plt.subplots(2,1)
# #fig.suptitle(' Flutter Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))

# ax[0].plot(time_int,qq[1,:],'k-',linewidth=2,label=r"$\theta$")
# ax[0].set_xlabel(r'$t \,\, [s]$')
# ax[0].set_ylabel(r'$\theta \,\, [RAD]$')

# ax[1].plot(time_int,qq[0,:],'k-',linewidth=2,label=r"$z$")
# ax[1].set_xlabel(r'$t \,\, [s]$')
# ax[1].set_ylabel(r'$z \,\, [m]$')
# #ax.set_xlim([-10,5])
# #ax.set_ylim([0,120])
# #ax.legend()
# plt.show()

# # ####
# # Xpitch = np.fft.fft(qq[0,:])
# # Xplunge = np.fft.fft(qq[1,:])
# # N = len(Xpitch)
# # n = np.arange(N)
# # T = N*dtau*chord/(2*v_inf)
# # freq = n/T 

# # plt.figure(figsize = (12, 6))
# # plt.subplot()

# # plt.stem(freq, Xpitch.real, 'b', \
# #         markerfmt="*-", basefmt="-b",label="pitch real")
# # plt.stem(freq, Xpitch.imag, 'b', \
# #         markerfmt="*-", basefmt="-b",label="pitch imag")
# # plt.stem(freq, Xplunge.real, 'k', \
# #          markerfmt="ko", basefmt="-b",label="plunge real")
# # plt.stem(freq, Xplunge.imag, 'k', \
# #          markerfmt="go", basefmt="-b",label="plunge imag")
# # plt.xlabel('Freq (Hz)')
# # plt.ylabel('FFT Amplitude |X(freq)|')
# # plt.xlim(0.0, 10.0)

# # plt.legend()
# # plt.show()

