"""
Volterra 2nd-order kernel identification via Laguerre basis
============================================================
Extends the existing Volterra identification framework to include
the FULL quadratic kernel H2 (m x m matrix), parameterised via
Laguerre polynomials.

Parameterisation
----------------
The full quadratic kernel H2 (m x m, symmetric) is written as:

    H2 = L @ C2 @ L.T

where:
    L  : (m, n_basis)   Laguerre basis matrix (same as for h1)
    C2 : (n_basis, n_basis)   coefficient matrix, SYMMETRIC

Only the upper-triangular half of C2 is stored as a design variable,
vech(C2), of length n_basis*(n_basis+1)//2.

The quadratic contribution to the output at sample k is:

    q_k = A_k @ H2 @ A_k.T  =  A_k @ (L @ C2 @ L.T) @ A_k.T
        = (A_k @ L) @ C2 @ (A_k @ L).T
        = z_k @ C2 @ z_k.T

where z_k = A_k @ L  is the Laguerre-projected input row (length n_basis).
This means we never need to form H2 explicitly during identification —
we work entirely in the compressed Laguerre space.

Full output equation per sample k:
    f_k = A_k @ h1  +  z_k @ C2 @ z_k.T

New components added here
--------------------------
1.  H2LaguerreParamComp     — reconstruct full H2 from vech(C2) for inspection
2.  H2QuadraticFeatureComp  — build the B matrix: b_k = vech-features of z_k ⊗ z_k
3.  H2ResidualComp          — compute the quadratic residual  r = f - A@h1 - B@vech(C2)
4.  H2RegularizationComp    — smoothness / ridge penalties on vech(C2)
5.  run_volterra_h2_identification() — top-level runner, mirrors run_volterra_identification()

Usage
-----
    from volterra_h2_laguerre import run_volterra_h2_identification

    # Stage 1 (optional but recommended): identify h1 first with your
    # existing run_volterra_identification(), then fix it.
    h_cl, h_cm, prob1 = run_volterra_identification(A, cl, cm, m, ...)

    # Stage 2: identify H2 on the residual, with h1 fixed
    H2_cl, H2_cm, prob2 = run_volterra_h2_identification(
        A, cl, cm, m,
        h1_cl=h_cl, h1_cm=h_cm,     # fix h1 from stage 1
        n_basis_h2=8,                # Laguerre basis size for H2
        alpha_laguerre=0.7,
        lambda_ridge_h2=1e-3,
        lambda_smooth_h2=1e-2,
        optimizer='SLSQP',
    )
"""

import numpy as np
import openmdao.api as om
from scipy.special import eval_laguerre
from itertools import combinations_with_replacement


# ════════════════════════════════════════════════════════════════════════
# UTILITY: LAGUERRE BASIS  (shared with existing LaguerreTransformComp)
# ════════════════════════════════════════════════════════════════════════

def build_laguerre_basis(m, n_basis, alpha):
    """
    Build the (m, n_basis) Laguerre basis matrix for one kernel segment.

    Parameters
    ----------
    m       : int    memory depth
    n_basis : int    number of Laguerre functions
    alpha   : float  Laguerre pole (0 < alpha < 1);
                     larger alpha = slower decay = better for slow systems

    Returns
    -------
    L : (m, n_basis) ndarray
    """
    t = np.arange(m, dtype=float)
    L = np.zeros((m, n_basis))
    scale = np.sqrt(1.0 - alpha**2)
    for n in range(n_basis):
        arg = -2.0 * alpha * t / (1.0 - alpha**2) if alpha < 1.0 else np.zeros(m)
        L[:, n] = scale * (alpha**t) * eval_laguerre(n, -arg)
    return L


# ════════════════════════════════════════════════════════════════════════
# UTILITY: VECH INDEXING
# ════════════════════════════════════════════════════════════════════════

def vech_indices(n):
    """Upper-triangular index pairs (i<=j) for an (n,n) symmetric matrix."""
    return list(combinations_with_replacement(range(n), 2))


def vech_to_full(v, n):
    """Reconstruct full symmetric (n,n) matrix from its vech."""
    idx = vech_indices(n)
    M = np.zeros((n, n))
    for (i, j), val in zip(idx, v):
        M[i, j] = val
        M[j, i] = val
    return M


def full_to_vech(M):
    n = M.shape[0]
    return np.array([M[i, j] for i, j in vech_indices(n)])


# ════════════════════════════════════════════════════════════════════════
# COMPONENT 1: QUADRATIC FEATURE MATRIX
# Build B such that  z_k @ C2 @ z_k.T  =  B_k @ vech(C2)
# where z_k = A_k @ L_h2   (the Laguerre-projected input)
# ════════════════════════════════════════════════════════════════════════

class H2QuadraticFeatureComp(om.ExplicitComponent):
    """
    Pre-compute the quadratic feature matrix B in the Laguerre space.

    For each sample k:
        z_k = A_k @ L_h2           shape (n_basis_h2,)
        B_k = vech-features of z_k  shape (n_basis_h2*(n_basis_h2+1)//2,)

    such that   z_k @ C2 @ z_k.T  =  B_k @ vech(C2)

    B does NOT depend on the design variables, so it is computed once
    during setup and stored.  This is the key efficiency gain.

    Parameters (options)
    ----------
    A          : (n, p) full regressor matrix
    m          : int    memory depth per input
    n_basis_h2 : int    Laguerre basis size for H2
    alpha      : float  Laguerre pole
    kernel_name: str
    """

    def initialize(self):
        self.options.declare('A',           types=np.ndarray)
        self.options.declare('m',           types=int)
        self.options.declare('n_basis_h2',  types=int,   default=8)
        self.options.declare('alpha',       types=float, default=0.7)
        self.options.declare('kernel_name', types=str)

    def setup(self):
        # B is a constant — no inputs needed, we output it directly
        A          = self.options['A']
        m          = self.options['m']
        n_basis_h2 = self.options['n_basis_h2']
        alpha      = self.options['alpha']
        name       = self.options['kernel_name']
        n, p       = A.shape

        # Build Laguerre basis  L_h2 : (m, n_basis_h2)
        # If A has multiple input blocks (e.g. pitch + zdot), each block
        # of m columns gets the SAME basis. We sum the projected blocks.
        n_blocks = p // m
        L_h2     = build_laguerre_basis(m, n_basis_h2, alpha)

        # Project each input block: Z_k = sum over blocks of  A_block_k @ L_h2
        # Result Z : (n, n_basis_h2)
        Z = np.zeros((n, n_basis_h2))
        for b in range(n_blocks):
            A_block = A[:, b*m:(b+1)*m]     # (n, m)
            Z      += A_block @ L_h2        # (n, n_basis_h2)

        # Build B : (n, n_vech)
        # B_k @ vech(C2)  =  z_k @ C2 @ z_k.T
        idx_pairs = vech_indices(n_basis_h2)
        n_vech    = len(idx_pairs)
        B         = np.empty((n, n_vech))
        for col, (i, j) in enumerate(idx_pairs):
            if i == j:
                B[:, col] = Z[:, i] * Z[:, j]
            else:
                B[:, col] = 2.0 * Z[:, i] * Z[:, j]

        # Store for use downstream
        self.B         = B
        self.Z         = Z
        self.L_h2      = L_h2
        self.idx_pairs = idx_pairs
        self.n_vech    = n_vech
        self.n          = n

        # This component has no OpenMDAO inputs — B is purely data
        # We expose it as a constant output so other components can use it
        # (In practice we pass B directly via options to H2ResidualComp)
        self.add_output(f'B_{name}', shape=(n, n_vech),
                        desc='Quadratic feature matrix in Laguerre space')

    def compute(self, inputs, outputs):
        name = self.options['kernel_name']
        outputs[f'B_{name}'] = self.B


# ════════════════════════════════════════════════════════════════════════
# COMPONENT 2: H2 RESIDUAL
# r = f - A @ h1 - B @ vech(C2)
# ════════════════════════════════════════════════════════════════════════

class H2ResidualComp(om.ExplicitComponent):
    """
    Compute residual for quadratic kernel identification:

        residual_h2 = f - linear_pred - B @ vech(C2)

    where:
        linear_pred = A @ h1   (fixed, passed as option or input)
        B           = quadratic feature matrix (constant, passed as option)
        vech(C2)    = design variable

    Parameters (options)
    ----------
    B              : (n, n_vech) quadratic feature matrix from H2QuadraticFeatureComp
    f              : (n,) output signal (Cl or Cm)
    linear_pred    : (n,) linear prediction A @ h1  (if h1 is fixed)
                     pass None to include h1 as an OpenMDAO input instead
    kernel_name    : str
    n_vech         : int   n_basis_h2*(n_basis_h2+1)//2
    h1_fixed       : bool  if True, linear_pred is a fixed array (option);
                           if False, residual also reads 'h1_pred' input
    """

    def initialize(self):
        self.options.declare('B',            types=np.ndarray)
        self.options.declare('f',            types=np.ndarray)
        self.options.declare('linear_pred',  default=None)
        self.options.declare('kernel_name',  types=str)
        self.options.declare('n_vech',       types=int)
        self.options.declare('h1_fixed',     default=True, types=bool)

    def setup(self):
        name    = self.options['kernel_name']
        n_vech  = self.options['n_vech']
        B       = self.options['B']
        n       = B.shape[0]

        # Design variable: vech(C2)
        self.add_input(f'vech_C2_{name}',    shape=n_vech,
                       desc='Upper-triangular Laguerre C2 coefficients')

        # If h1 is not fixed, we also accept the linear prediction as input
        if not self.options['h1_fixed']:
            self.add_input(f'linear_pred_{name}', shape=n,
                           desc='Linear prediction A @ h1')

        self.add_output(f'residual_h2_{name}', shape=n,
                        desc='Quadratic residual')

        # Jacobian w.r.t. vech_C2 is -B (constant)
        self.declare_partials(f'residual_h2_{name}', f'vech_C2_{name}',
                              val=-B)

        if not self.options['h1_fixed']:
            self.declare_partials(f'residual_h2_{name}',
                                   f'linear_pred_{name}',
                                   val=-np.eye(n))

    def compute(self, inputs, outputs):
        name        = self.options['kernel_name']
        B           = self.options['B']
        f           = self.options['f']
        h1_fixed    = self.options['h1_fixed']

        vech_C2     = inputs[f'vech_C2_{name}']
        quad_pred   = B @ vech_C2

        if h1_fixed:
            lin_pred = self.options['linear_pred']
        else:
            lin_pred = inputs[f'linear_pred_{name}']

        outputs[f'residual_h2_{name}'] = f - lin_pred - quad_pred


# ════════════════════════════════════════════════════════════════════════
# COMPONENT 3: H2 REGULARIZATION
# Smoothness (second derivative on C2 columns) + ridge on vech(C2)
# ════════════════════════════════════════════════════════════════════════

class H2RegularizationComp(om.ExplicitComponent):
    """
    Regularization penalties on the H2 Laguerre coefficients vech(C2).

    Two penalties:
    1. Ridge:      lambda_ridge * ||vech(C2)||^2
    2. Smoothness: lambda_smooth * ||D2 @ vech(C2)||^2
       where D2 is a second-difference operator along the vech vector
       (penalises rapid variation between adjacent C2 entries)

    Symmetry enforcement is implicit: since we only identify vech(C2)
    (upper triangle), C2 is symmetric by construction.
    """

    def initialize(self):
        self.options.declare('kernel_name',    types=str)
        self.options.declare('n_vech',         types=int)
        self.options.declare('lambda_ridge',   default=1e-3)
        self.options.declare('lambda_smooth',  default=1e-2)

    def setup(self):
        name   = self.options['kernel_name']
        n_vech = self.options['n_vech']

        self.add_input(f'vech_C2_{name}',      shape=n_vech)
        self.add_output(f'ridge_h2_{name}',    val=0.0)
        self.add_output(f'smooth_h2_{name}',   val=0.0)

        # Second-difference operator on the vech vector
        if n_vech > 2:
            self.D2 = np.diff(np.eye(n_vech), n=2, axis=0)   # (n_vech-2, n_vech)
            self.D2TD2 = self.D2.T @ self.D2
        else:
            self.D2    = np.zeros((max(1, n_vech-2), n_vech))
            self.D2TD2 = np.zeros((n_vech, n_vech))

        self.declare_partials(f'ridge_h2_{name}',  f'vech_C2_{name}')
        self.declare_partials(f'smooth_h2_{name}', f'vech_C2_{name}')

    def compute(self, inputs, outputs):
        name          = self.options['kernel_name']
        lam_r         = self.options['lambda_ridge']
        lam_s         = self.options['lambda_smooth']
        v             = inputs[f'vech_C2_{name}']

        outputs[f'ridge_h2_{name}']  = lam_r * np.dot(v, v)
        D2v = self.D2 @ v
        outputs[f'smooth_h2_{name}'] = lam_s * np.dot(D2v, D2v)

    def compute_partials(self, inputs, partials):
        name  = self.options['kernel_name']
        lam_r = self.options['lambda_ridge']
        lam_s = self.options['lambda_smooth']
        v     = inputs[f'vech_C2_{name}']

        partials[f'ridge_h2_{name}',  f'vech_C2_{name}'] = 2 * lam_r * v
        partials[f'smooth_h2_{name}', f'vech_C2_{name}'] = 2 * lam_s * (self.D2TD2 @ v)


# ════════════════════════════════════════════════════════════════════════
# COMPONENT 4: H2 OBJECTIVE
# Adds the quadratic residual norm and H2 penalties to the objective
# ════════════════════════════════════════════════════════════════════════

class H2ObjectiveComp(om.ExplicitComponent):
    """
    Objective for the H2 identification stage:

        obj = sum_name ( ||residual_h2_name||^2
                       + ridge_h2_name
                       + smooth_h2_name )
    """

    def initialize(self):
        self.options.declare('kernel_names', types=list)
        self.options.declare('n_samples',    types=int)

    def setup(self):
        names    = self.options['kernel_names']
        n        = self.options['n_samples']

        for name in names:
            self.add_input(f'residual_h2_{name}', shape=n)
            self.add_input(f'ridge_h2_{name}',    val=0.0)
            self.add_input(f'smooth_h2_{name}',   val=0.0)

        self.add_output('objective_h2', val=0.0)

        for name in names:
            self.declare_partials('objective_h2', f'residual_h2_{name}')
            self.declare_partials('objective_h2', f'ridge_h2_{name}',  val=1.0)
            self.declare_partials('objective_h2', f'smooth_h2_{name}', val=1.0)

    def compute(self, inputs, outputs):
        names = self.options['kernel_names']
        obj   = 0.0
        for name in names:
            r    = inputs[f'residual_h2_{name}']
            obj += np.dot(r, r)
            obj += inputs[f'ridge_h2_{name}']
            obj += inputs[f'smooth_h2_{name}']
        outputs['objective_h2'] = obj

    def compute_partials(self, inputs, partials):
        names = self.options['kernel_names']
        for name in names:
            r = inputs[f'residual_h2_{name}']
            partials['objective_h2', f'residual_h2_{name}'] = 2.0 * r


# ════════════════════════════════════════════════════════════════════════
# COMPONENT 5: RECONSTRUCTION — vech(C2) → full H2  (for inspection)
# ════════════════════════════════════════════════════════════════════════

class H2ReconstructComp(om.ExplicitComponent):
    """
    Reconstruct the full (m, m) kernel H2 from vech(C2):

        H2 = L_h2 @ C2 @ L_h2.T

    This component is NOT part of the optimisation — it is used only
    after convergence to obtain the physical kernel matrix for plotting
    and for use in your aeroelastic solver.
    """

    def initialize(self):
        self.options.declare('L_h2',       types=np.ndarray,
                             desc='Laguerre basis (m, n_basis_h2)')
        self.options.declare('kernel_name', types=str)
        self.options.declare('n_vech',      types=int)

    def setup(self):
        name   = self.options['kernel_name']
        L_h2   = self.options['L_h2']
        m, nb  = L_h2.shape
        n_vech = self.options['n_vech']

        self.add_input(f'vech_C2_{name}', shape=n_vech)
        self.add_output(f'H2_{name}',     shape=(m, m),
                        desc='Full quadratic kernel matrix')

        self.declare_partials(f'H2_{name}', f'vech_C2_{name}')

    def compute(self, inputs, outputs):
        name   = self.options['kernel_name']
        L_h2   = self.options['L_h2']
        n_vech = self.options['n_vech']
        nb     = L_h2.shape[1]

        v      = inputs[f'vech_C2_{name}']
        C2     = vech_to_full(v, nb)
        H2     = L_h2 @ C2 @ L_h2.T
        outputs[f'H2_{name}'] = H2

    def compute_partials(self, inputs, partials):
        name   = self.options['kernel_name']
        L_h2   = self.options['L_h2']
        n_vech = self.options['n_vech']
        nb     = L_h2.shape[1]
        m      = L_h2.shape[0]

        # dH2/d(vech_C2): shape (m*m, n_vech)
        idx    = vech_indices(nb)
        dH2    = np.zeros((m*m, n_vech))
        for col, (i, j) in enumerate(idx):
            # d(L C2 L.T)/dC2_ij = L[:,i:i+1] @ L[:,j:j+1].T  (+ symmetric part)
            dC2 = np.zeros((nb, nb))
            dC2[i, j] = 1.0
            dC2[j, i] = 1.0 if i != j else 0.0   # symmetrize
            dH2_ij = L_h2 @ dC2 @ L_h2.T
            dH2[:, col] = dH2_ij.ravel()

        partials[f'H2_{name}', f'vech_C2_{name}'] = dH2


# ════════════════════════════════════════════════════════════════════════
# OPENMDAO GROUP — H2 IDENTIFICATION STAGE
# ════════════════════════════════════════════════════════════════════════

class VoltterraH2Group(om.Group):
    """
    OpenMDAO group for the H2 identification stage.

    Assumes h1_cl and h1_cm are already known (fixed as constants).
    The design variables are vech_C2_cl and vech_C2_cm.
    """

    def initialize(self):
        self.options.declare('A',             types=np.ndarray)
        self.options.declare('cl',            types=np.ndarray)
        self.options.declare('cm',            types=np.ndarray)
        self.options.declare('m',             types=int)
        self.options.declare('h1_cl',         types=np.ndarray)
        self.options.declare('h1_cm',         types=np.ndarray)
        self.options.declare('n_basis_h2',    types=int,   default=8)
        self.options.declare('alpha',         types=float, default=0.7)
        self.options.declare('lambda_ridge',  types=float, default=1e-3)
        self.options.declare('lambda_smooth', types=float, default=1e-2)

    def setup(self):
        A            = self.options['A']
        cl           = self.options['cl']
        cm           = self.options['cm']
        m            = self.options['m']
        h1_cl        = self.options['h1_cl']
        h1_cm        = self.options['h1_cm']
        n_basis_h2   = self.options['n_basis_h2']
        alpha        = self.options['alpha']
        lam_r        = self.options['lambda_ridge']
        lam_s        = self.options['lambda_smooth']

        n, p         = A.shape
        kernel_names = ['cl', 'cm']
        n_vech       = n_basis_h2 * (n_basis_h2 + 1) // 2

        # ── fixed linear predictions ──────────────────────────────────
        lin_pred_cl = A @ h1_cl
        lin_pred_cm = A @ h1_cm

        # ── build quadratic feature matrices (constant) ───────────────
        feat_cl = H2QuadraticFeatureComp(A=A, m=m, n_basis_h2=n_basis_h2,
                                          alpha=alpha, kernel_name='cl')
        feat_cm = H2QuadraticFeatureComp(A=A, m=m, n_basis_h2=n_basis_h2,
                                          alpha=alpha, kernel_name='cm')
        # Run compute manually to extract B (no OpenMDAO dependency needed)
        prob_tmp = om.Problem()
        prob_tmp.model.add_subsystem('f', feat_cl)
        prob_tmp.setup(); prob_tmp.run_model()
        B_cl = prob_tmp.get_val('f.B_cl').reshape(n, n_vech)

        prob_tmp2 = om.Problem()
        prob_tmp2.model.add_subsystem('f', feat_cm)
        prob_tmp2.setup(); prob_tmp2.run_model()
        B_cm = prob_tmp2.get_val('f.B_cm').reshape(n, n_vech)

        # Store L_h2 for reconstruction later
        self._L_h2   = build_laguerre_basis(m, n_basis_h2, alpha)
        self._n_vech = n_vech
        self._B      = {'cl': B_cl, 'cm': B_cm}

        # ── IndepVarComp for design variables ─────────────────────────
        ivcomp = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        for name in kernel_names:
            ivcomp.add_output(f'vech_C2_{name}', val=np.zeros(n_vech))

        # ── residual components ───────────────────────────────────────
        for name, f_arr, lin_pred, B_mat in [
            ('cl', cl, lin_pred_cl, B_cl),
            ('cm', cm, lin_pred_cm, B_cm),
        ]:
            self.add_subsystem(
                f'residual_h2_{name}',
                H2ResidualComp(B=B_mat, f=f_arr,
                                linear_pred=lin_pred,
                                kernel_name=name,
                                n_vech=n_vech,
                                h1_fixed=True),
                promotes_inputs=[f'vech_C2_{name}'],
                promotes_outputs=[f'residual_h2_{name}']
            )

        # ── regularization components ─────────────────────────────────
        for name in kernel_names:
            self.add_subsystem(
                f'reg_h2_{name}',
                H2RegularizationComp(kernel_name=name, n_vech=n_vech,
                                      lambda_ridge=lam_r,
                                      lambda_smooth=lam_s),
                promotes_inputs=[f'vech_C2_{name}'],
                promotes_outputs=[f'ridge_h2_{name}', f'smooth_h2_{name}']
            )

        # ── objective ─────────────────────────────────────────────────
        self.add_subsystem(
            'objective',
            H2ObjectiveComp(kernel_names=kernel_names, n_samples=n),
            promotes_inputs=(
                [f'residual_h2_{nm}' for nm in kernel_names] +
                [f'ridge_h2_{nm}'    for nm in kernel_names] +
                [f'smooth_h2_{nm}'   for nm in kernel_names]
            ),
            promotes_outputs=['objective_h2']
        )


# ════════════════════════════════════════════════════════════════════════
# TOP-LEVEL RUNNER
# ════════════════════════════════════════════════════════════════════════

def run_volterra_h2_identification(
        A, cl, cm, m,
        h1_cl, h1_cm,
        n_basis_h2   = 8,
        alpha_laguerre = 0.7,
        lambda_ridge   = 1e-3,
        lambda_smooth  = 1e-2,
        vech_bound     = 500.0,
        optimizer      = 'SLSQP',
        max_iter       = 2000,
        tol            = 1e-8,
        verbose        = True):
    """
    Identify the full quadratic Volterra kernel H2 via Laguerre basis.

    Parameters
    ----------
    A              : (n, p)   regressor matrix (same as used for h1)
    cl, cm         : (n,)     training outputs
    m              : int      memory depth per input block
    h1_cl, h1_cm   : (p,)    already-identified first-order kernels
    n_basis_h2     : int      number of Laguerre functions for H2
                              (typical: 6-12; H2 unknowns = n_basis_h2*(n_basis_h2+1)/2)
    alpha_laguerre : float    Laguerre pole (0 < alpha < 1)
    lambda_ridge   : float    L2 penalty on vech(C2)
    lambda_smooth  : float    second-derivative penalty on vech(C2)
    vech_bound     : float    symmetric box bound on vech(C2) entries
    optimizer      : str      'SLSQP' or 'IPOPT'
    max_iter       : int
    tol            : float
    verbose        : bool

    Returns
    -------
    H2_cl  : (m, m) ndarray   full quadratic kernel for Cl
    H2_cm  : (m, m) ndarray   full quadratic kernel for Cm
    vech_cl: (n_vech,)        Laguerre C2 vech coefficients for Cl
    vech_cm: (n_vech,)        Laguerre C2 vech coefficients for Cm
    prob   : om.Problem       OpenMDAO problem (for inspection)
    """
    n, p     = A.shape
    n_vech   = n_basis_h2 * (n_basis_h2 + 1) // 2

    if verbose:
        print("="*70)
        print("VOLTERRA H2 IDENTIFICATION — LAGUERRE BASIS")
        print("="*70)
        print(f"  n={n}, p={p}, m={m}")
        print(f"  n_basis_h2={n_basis_h2}  =>  {n_vech} unknowns per output")
        print(f"  alpha={alpha_laguerre}  lambda_ridge={lambda_ridge}  "
              f"lambda_smooth={lambda_smooth}")
        print(f"  Ratio n / n_vech = {n / n_vech:.1f}  "
              f"({'OK' if n/n_vech > 5 else 'WARNING: consider reducing n_basis_h2'})")

    prob = om.Problem()
    prob.model = VoltterraH2Group(
        A=A, cl=cl, cm=cm, m=m,
        h1_cl=h1_cl, h1_cm=h1_cm,
        n_basis_h2=n_basis_h2,
        alpha=alpha_laguerre,
        lambda_ridge=lambda_ridge,
        lambda_smooth=lambda_smooth,
    )

    # driver
    if optimizer == 'SLSQP':
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['maxiter']   = max_iter
        prob.driver.options['tol']       = tol
    elif optimizer == 'IPOPT':
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer']        = 'IPOPT'
        prob.driver.opt_settings['max_iter']    = max_iter
        prob.driver.opt_settings['tol']         = tol
    else:
        raise ValueError(f"Optimizer '{optimizer}' not supported. "
                         "Use 'SLSQP' or 'IPOPT'.")

    # design variables and objective
    prob.model.add_design_var('vech_C2_cl', lower=-vech_bound, upper=vech_bound)
    prob.model.add_design_var('vech_C2_cm', lower=-vech_bound, upper=vech_bound)
    prob.model.add_objective('objective_h2')

    prob.setup()

    # warm-start: least-squares solution in Laguerre space as initial guess
    # (much better than zeros for the optimizer)
    group = prob.model
    B_cl  = group._B['cl']
    B_cm  = group._B['cm']
    lin_pred_cl = A @ h1_cl
    lin_pred_cm = A @ h1_cm
    res_cl = cl - lin_pred_cl
    res_cm = cm - lin_pred_cm

    v0_cl, _, _, _ = np.linalg.lstsq(B_cl, res_cl, rcond=None)
    v0_cm, _, _, _ = np.linalg.lstsq(B_cm, res_cm, rcond=None)
    v0_cl = np.clip(v0_cl, -vech_bound, vech_bound)
    v0_cm = np.clip(v0_cm, -vech_bound, vech_bound)

    prob.set_val('vech_C2_cl', v0_cl)
    prob.set_val('vech_C2_cm', v0_cm)

    if verbose:
        r2_lin_cl = 1 - np.var(res_cl)/np.var(cl)
        r2_lin_cm = 1 - np.var(res_cm)/np.var(cm)
        r2_0_cl   = 1 - np.var(res_cl - B_cl@v0_cl)/np.var(cl)
        r2_0_cm   = 1 - np.var(res_cm - B_cm@v0_cm)/np.var(cm)
        print(f"\n  Linear-only R²:  Cl={1-np.var(res_cl)/np.var(cl):.4f}  "  # noqa
              f"Cm={1-np.var(res_cm)/np.var(cm):.4f}")
        print(f"  LS warm-start R²: Cl={r2_0_cl:.4f}  Cm={r2_0_cm:.4f}")
        print("\n  Running optimizer ...\n")

    prob.run_driver()

    # extract results
    vech_cl = prob.get_val('vech_C2_cl').copy()
    vech_cm = prob.get_val('vech_C2_cm').copy()

    L_h2   = group._L_h2
    nb     = n_basis_h2
    H2_cl  = L_h2 @ vech_to_full(vech_cl, nb) @ L_h2.T
    H2_cm  = L_h2 @ vech_to_full(vech_cm, nb) @ L_h2.T

    if verbose:
        res_cl_final = cl  - lin_pred_cl - B_cl @ vech_cl
        res_cm_final = cm  - lin_pred_cm - B_cm @ vech_cm
        r2_final_cl  = 1 - np.var(res_cl_final)/np.var(cl)
        r2_final_cm  = 1 - np.var(res_cm_final)/np.var(cm)
        nmse_cl      = np.mean(res_cl_final**2)/np.var(cl)
        nmse_cm      = np.mean(res_cm_final**2)/np.var(cm)
        print("\n" + "="*70)
        print("H2 IDENTIFICATION RESULTS")
        print("="*70)
        print(f"  Final objective  : {prob.get_val('objective_h2')[0]:.4e}")
        print(f"  Total R²  Cl     : {r2_final_cl:.5f}   Cm: {r2_final_cm:.5f}")
        print(f"  NMSE      Cl     : {nmse_cl:.4e}   Cm: {nmse_cm:.4e}")
        print(f"  ||H2_cl||_F      : {np.linalg.norm(H2_cl):.4e}")
        print(f"  ||H2_cm||_F      : {np.linalg.norm(H2_cm):.4e}")

    return H2_cl, H2_cm, vech_cl, vech_cm, prob


# ════════════════════════════════════════════════════════════════════════
# PREDICTION HELPER (use after identification)
# ════════════════════════════════════════════════════════════════════════

def predict_volterra_order2(A, h1, H2):
    """
    Evaluate the first + second order Volterra prediction:

        f_pred_k = A_k @ h1  +  A_k @ H2 @ A_k.T

    Efficient: uses  diag(A @ H2 @ A.T) = sum((A @ H2) * A, axis=1)
    to avoid forming the full (n, n) outer product matrix.

    Parameters
    ----------
    A  : (n, p)
    h1 : (p,)
    H2 : (p, p)  symmetric

    Returns
    -------
    f_pred : (n,)
    """
    linear_part    = A @ h1
    quadratic_part = np.sum((A @ H2) * A, axis=1)
    return linear_part + quadratic_part