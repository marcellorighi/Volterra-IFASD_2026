import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
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

Mach = 0.8      # freestream Mach number
AOA = 0    # wind off angle-of-attack [deg]

q = 7000. # 1 deg: 8245.5 

v_inf = 131.41  # freestream velocity from CFD simulation [m/s]
rho_inf = 0.832 # freestream density from CFD simulation [kg/m^3]
q_inf = 0.5*rho_inf*v_inf**2 
delta_t = 0.0002 # transient time step [seconds]
chord = 0.41

dtau = delta_t * v_inf /(2 * chord)
resample_steps = 4
m = 80

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

heave_input_int = - gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
heave_input_grad = np.gradient(heave_input_int,time_vector)
heave_input_grad_resampled = heave_input_grad[::resample_steps]/v_inf

pitch_input_grad = np.gradient(gen_disp[1,:]*29.85872 /180*np.pi,time_vector)
pitch_input_grad_resampled = pitch_input_grad[::resample_steps]/v_inf


cl_output = gen_force[0,:] / 0.106654 / (chord*chord*2)
cl_output_resampled = gen_force[0,::resample_steps] / 0.106654 / (chord*chord*2)

cm_output = gen_force[1,:]/ (29.86*np.pi/180) / (chord*chord*chord*2)
cm_output_resampled = gen_force[1,::resample_steps] / (29.86*np.pi/180) / (chord*chord*chord*2)

cm_0 = gen_static_forces[1] / (29.86*np.pi/180) / (chord*chord*chord*2)
cl_0 = gen_static_forces[0] / 0.106654 / (chord*chord*2)

print("Cm_0 and Cl_0", cm_0, cl_0)
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

plt.plot(time_vector_resampled, heave_input_grad_resampled*180/np.pi,label="heave input")
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
A_zdot = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn]
    A_thetadot[icolumn:,icolumn] = pitch_input_grad[:ntsteps-icolumn]
    A_zdot[icolumn:,icolumn] = heave_input_grad[:ntsteps-icolumn]

A = np.hstack((A_pitch,A_thetadot,A_zdot))

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
A_zdot2 = A_zdot*A_zdot
A_pitch_zdot = A_pitch*A_zdot
A_thetadot_zdot = A_thetadot*A_zdot

# A2 = np.hstack( (np.hstack((A_pitch2,A_thetadot2,A_heave2)),(A_pitch_heave,A_thetadot_heave)) )
A2 = np.hstack( (A_pitch2,A_thetadot2,A_zdot2,A_pitch_zdot,A_thetadot_zdot)) 

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

pitch_input           = 1e0*gen_disp[1,:]*29.85872 /180*np.pi # pitch input if you want to convert to deg
pitch_input_resampled = 1e0*gen_disp[1,::resample_steps]*29.85872 /180*np.pi # pitch input if you want to convert to deg

ntsteps            = len(pitch_input)
ntsteps_resampled  = len(pitch_input_resampled)

time_vector           = np.linspace(0.,ntsteps*delta_t, ntsteps)
time_vector_resampled = time_vector[::resample_steps] #np.linspace(0.,resample_steps*delta_t*ntsteps_resampled,ntsteps_resampled)

tau_vector = time_vector*2*v_inf/chord
tau_vector_resampled = time_vector_resampled*2*v_inf/chord

heave_input_int = - gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
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
A_thetadot = np.zeros((ntsteps,m))
A_zdot = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn]
    A_thetadot[icolumn:,icolumn] = pitch_input_grad[:ntsteps-icolumn]
    A_zdot[icolumn:,icolumn] = heave_input_grad[:ntsteps-icolumn]

A = np.hstack((A_pitch,A_thetadot,A_zdot))

A_pitch2 = A_pitch*A_pitch
A_thetadot2 = A_thetadot*A_thetadot
A_zdot2 = A_zdot*A_zdot
A_pitch_zdot = A_pitch*A_zdot
A_thetadot_zdot = A_thetadot*A_zdot

A2 = np.hstack( (A_pitch2,A_thetadot2,A_zdot2,A_pitch_zdot,A_thetadot_zdot)) 

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



# 2DOF SS system 

mass= 87; 
Iy= 3.765; 
kz=mass*(3.33*2*np.pi)**2; 
ktheta=Iy*(5.20*2*np.pi)**2; 

print("k_z, k_theta: ", kz, ktheta)

chord = 0.4032
area = 0.3303 

xi = 0.00

n_dof = 2 # physical DOFs

# K_aa = np.array([[ktheta, 0],[0, kz]])
# M_aa = np.array([[Iy, 0],[0,mass]])
K_aa = np.array([[kz, 0],[0, ktheta]])
M_aa = np.array([[mass, 0],[0,Iy]])
C_aa = 2*xi* np.diag([np.sqrt(Iy*ktheta), np.sqrt(mass*kz)])

ndof_aero = 3*(m-1) 
ndof = ndof_aero + 2 * n_dof 

# AA = np.zeros((ndof,ndof))
# BB = np.zeros((ndof,ndof))

# Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
# Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])


DT = delta_t*resample_steps
dtau = delta_t*(2*v_inf)/chord * resample_steps

    
# aerodynamic forces contribution by (n+1) aoa_eff

beta = 0.5 

Q_lift = q * area 
Q_ma = q * area * chord 

# Volterra coeffs 

h_cl_theta = theta_cl[:m]
h_cl_thetadot = theta_cl[m:2*m]
h_cl_zdot = theta_cl[2*m:]

h_cm_theta = theta_cm[:m]
h_cm_thetadot = theta_cm[m:2*m]
h_cm_zdot = theta_cm[2*m:]

print("coeffs size",theta_cl.shape,h_cl_theta.shape,h_cl_thetadot.shape,h_cl_zdot.shape)
# prepare AA and BB 

n_dof = 2 

AA = np.zeros((ndof,ndof))
BB = np.zeros((ndof,ndof))

Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

Aphys_D = Bphys + DT*(1.-beta)*Aphys
Bphys_D = Bphys - DT*beta*Aphys 

AA[0:2*n_dof,0:2*n_dof] = Aphys_D 
BB[0:2*n_dof,0:2*n_dof] = Bphys_D 

BB[4:,4:] = np.eye(3*(m-1))

# TEMP

# h_cl_z_dot = np.zeros(m)
# h_cm_z_dot = np.zeros(m)

# Volterra coefficients at time (n+1)

BB[2,1] -= DT * Q_lift * h_cl_theta[0] * beta 
BB[2,2] -= - DT * Q_lift * h_cl_zdot[0] * beta / v_inf
BB[2,3] -= DT * Q_lift * h_cl_thetadot[0] * beta / v_inf

BB[3,1] -= DT * Q_ma * h_cm_theta[0] * beta 
BB[3,2] -= - DT * Q_ma * h_cm_zdot[0] * beta / v_inf
BB[3,3] -= DT * Q_ma * h_cm_thetadot[0] * beta / v_inf

# Volterra coefficients at time (n)

AA[2,1] += DT * Q_lift * h_cl_theta[1] * beta
AA[2,2] += - DT * Q_lift * h_cl_zdot[1] * beta / v_inf
AA[2,3] += DT * Q_lift * h_cl_thetadot[1] * beta / v_inf 

AA[3,1] += DT * Q_ma * h_cm_theta[1] * beta
AA[3,2] += - DT * Q_ma * h_cm_zdot[1] * beta / v_inf
AA[3,3] += DT * Q_ma * h_cm_thetadot[1] * beta / v_inf

AA[2,1] += DT * Q_lift * h_cl_theta[0] * (1-beta) 
AA[2,2] += - DT * Q_lift * h_cl_zdot[0] * (1-beta) / v_inf
AA[2,3] += DT * Q_lift * h_cl_thetadot[0] * (1-beta) / v_inf

AA[3,1] += DT * Q_ma * h_cm_theta[0] * (1-beta) 
AA[3,2] += - DT * Q_ma * h_cm_zdot[0] * (1-beta) / v_inf
AA[3,3] += DT * Q_ma * h_cm_thetadot[0] * (1-beta) / v_inf

# Volterra coefficients at times (n-1 -> n-m+1)

icolumn1_theta = 4
icolumn2_theta = icolumn1_theta + (m-2)

icolumn1_thetadot = 4 + (m-1)
icolumn2_thetadot = icolumn1_thetadot + (m-2)

icolumn1_heavedot = 4 + 2 * (m-1)
icolumn2_heavedot = icolumn1_heavedot + (m-2)

AA[2,icolumn1_theta:icolumn2_theta] = DT * Q_lift * h_cl_theta[2:] * beta 
AA[2,icolumn1_heavedot:icolumn2_heavedot] = DT * Q_lift * h_cl_zdot[2:] * beta 
AA[2,icolumn1_thetadot:icolumn2_thetadot] = DT * Q_lift * h_cl_thetadot[2:] * beta 

AA[3,icolumn1_theta:icolumn2_theta] = DT * Q_ma * h_cm_theta[2:] * beta 
AA[3,icolumn1_heavedot:icolumn2_heavedot] = DT * Q_ma * h_cm_zdot[2:] * beta 
AA[3,icolumn1_thetadot:icolumn2_thetadot] = DT * Q_ma * h_cm_thetadot[2:] * beta 

# Volterra coefficients at times (n-1 -> n-m)

icolumn1_theta = 4
icolumn2_theta = icolumn1_theta + (m-1)

icolumn1_thetadot = 4 + (m-1)
icolumn2_thetadot = icolumn1_thetadot + (m-1)

icolumn1_heavedot = 4 + 2 * (m-1)
icolumn2_heavedot = icolumn1_heavedot + (m-1)

AA[2,icolumn1_theta:icolumn2_theta] += DT * Q_lift * h_cl_theta[1:] * (1-beta) 
AA[2,icolumn1_heavedot:icolumn2_heavedot] += DT * Q_lift * h_cl_zdot[1:] * (1-beta) 
AA[2,icolumn1_thetadot:icolumn2_thetadot] += DT * Q_lift * h_cl_thetadot[1:] * (1-beta) 

AA[3,icolumn1_theta:icolumn2_theta] += DT * Q_ma * h_cm_theta[1:] * (1-beta) 
AA[3,icolumn1_heavedot:icolumn2_heavedot] += DT * Q_ma * h_cm_zdot[1:] * (1-beta) 
AA[3,icolumn1_thetadot:icolumn2_thetadot] += DT * Q_ma * h_cm_thetadot[1:] * (1-beta) 

# update aerodynamic states

for i_dof in range(3):     
    irow1 = 4+1 + i_dof * (m-1)
    irow2 = irow1 + m -1 - 1 
    icolumn1 = irow1 - 1
    icolumn2 = irow2 - 1 #icolumn1 + m -1 
    print("update coeffs indices ",irow1,irow2,icolumn1,icolumn2)
    AA[irow1:irow2, icolumn1:icolumn2] = np.eye(m-2)

AA[4,1] = 1.
AA[4 + (m-1),3] = 1./v_inf #THETA DOT ???
AA[4 + 2*(m-1),2] = - 1./v_inf #Z DOT??? 

# print("row 5",AA[4,:])
# print("row 6",AA[5,:])
# print("row 7",AA[6,:])

# time integration 

nt_int = 16*1024 # numerical integration steps 

tau_int = np.linspace(0,nt_int*dtau,nt_int)
time_int = np.linspace(0,nt_int*DT,nt_int)

qq = np.zeros((ndof,nt_int))

qping = 1.0 # heave 
qq[2,0] = qping # PING
qq[3,0] = qping # PING

fNL = np.zeros(ndof)

BB_1 = linalg.inv(BB)

f_0 = np.zeros(4 + 3*(m-1))
f_0[2] = Q_lift * cl_0 
f_0[3] = Q_ma * cm_0

theta_lags = np.zeros(m)
thetadot_lags = np.zeros(m)
zdot_lags = np.zeros(m)

for it in range(1,nt_int):
    theta_lags[0] = qq[1,it-1]
    thetadot_lags[0] = qq[3,it-1]/v_inf
    zdot_lags[0] = -qq[2,it-1]/v_inf

    theta_lags[1:] = qq[4:4+m-1,it-1]
    thetadot_lags[1:] = qq[4 + m-1: 4+(m-1)*2,it-1]
    zdot_lags[1:] = qq[4 + 2*(m-1):,it-1]

    all_lags_1 = np.concatenate([theta_lags, thetadot_lags, zdot_lags, theta_lags, thetadot_lags])
    all_lags_2 = np.concatenate([theta_lags, thetadot_lags, zdot_lags, zdot_lags, zdot_lags])

    cl_NL = np.dot(theta_cl2, all_lags_1*all_lags_2)
    cm_NL = np.dot(theta_cm2, all_lags_1*all_lags_2)

    # print(f_NL_lift, f_NL_ma)

    f_0[2] = Q_lift * (cl_0 + cl_NL)
    f_0[3] = Q_ma * (cm_0 + cm_NL)
 
    # if flutter_study == 'NL':       
    #     fNL[2] = q*area*chord*(kernels_cm_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
    #     fNL[3] = q*area*(kernels_cl_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
    #     qq[:,it] = AA@(qq[:,it-1] + DT*fNL) 
    # if flutter_study == 'lin':
    qq[:,it] = BB_1@(AA@qq[:,it-1] + DT*f_0) 

    # print(qq[4+m:4+m*3,it])
    #qq[:,it] = AA@qq[:,it-1] 

# fig, ax = plt.subplots(2,1)
# #fig.suptitle(' Flutter Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))

# ax[0].plot(time_int,qq[1,:]*180/np.pi,'k-',linewidth=2,label=r"$\theta$")
# ax[0].set_xlabel(r'$t \,\, [s]$')
# ax[0].set_ylabel(r'$\theta \,\, [DEG]$')

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


fig, ax = plt.subplots(2, 1, figsize=(10, 8))

plt.subplot(2, 1, 1)

plt.plot(time_int,qq[0,:],'k-',linewidth=2,label=r"$\theta$")
plt.xlabel(r'$t\,[s]$')
plt.ylabel(r'$z\, [m]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)

plt.plot(time_int,qq[1,:]*180/np.pi,'k-',linewidth=2,label=r"$\theta$")
plt.xlabel(r'$t\,[s]$')
plt.ylabel(r'$\theta\, [DEG]$')
plt.legend()
plt.grid(True)

plt.tight_layout() 
plt.show()

