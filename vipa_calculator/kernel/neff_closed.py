"""Strip coherence n_eff in closed form (calculation R54 of the main document).

Library module (no command-line use, no files). n_eff_closed(...) returns (n_eff, mu, D_psi(c)); the other functions
are its building blocks. Lengths in m, wl in m.

Structure function of the height (isotropic power law f^-alpha, variance 1, lower limit 1/D_m, r << D_m):
    D_h(r) = C_a (r/D_m)^nu,  nu = alpha-2,  C_a = 2 (alpha-2) (2 pi)^(alpha-2) I(alpha),  I(alpha) = 2^(-2H-1) Gamma(1-H)/(H Gamma(1+H)), H = nu/2
Accumulated phase over k round trips = line mean over L = k dx; structure function of two strips at distance u (continuum):
    D_psi(u) = (2 k0 sigma)^2 k^2 (2/L^2) int_0^L (L-s) [D_h(sqrt(s^2+u^2)) - D_h(s)] ds       (closed form: 2F1)
Power-law approximation D_psi(u) ~ D_psi(c) (u/c)^mu, mu = dlnD/dlnu at u = c/2; then for the uniformly illuminated chord c
    A^2 = (2/c^2) int_0^c (c-u) exp(-D_psi(u)/2) du = (2/c^2)[ c l gamma(1/mu, z)/mu - l^2 gamma(2/mu, z)/mu ],  l = c (2/D_psi(c))^(1/mu), z = (c/l)^mu
    n_eff = 1/A^2.  Fine band: D += 2 (2 k0 sigma_fein)^2 k (independent strips).
"""
import numpy as np
from scipy.special import gamma, gammainc, hyp2f1
def I_alpha(a):
    """Integral I(alpha) of the structure-function constant (see module docstring)."""
    H=(a-2)/2; return 2**(-2*H-1)*gamma(1-H)/(H*gamma(1+H))
# constant C_a of the structure function D_h(r) = C_a (r/D_m)^(alpha-2)
def C_alpha(a): return 2*(a-2)*(2*np.pi)**(a-2)*I_alpha(a)
def J(L,u,nu):
    """int_0^L (L-s)[(s^2+u^2)^(nu/2) - s^nu] ds, closed form."""
    # int_0^L (s^2+u^2)^(nu/2) ds = L u^nu 2F1(-nu/2,1/2;3/2;-L^2/u^2); int_0^L s (s^2+u^2)^(nu/2) ds = [(L^2+u^2)^(nu/2+1) - u^(nu+2)]/(nu+2)
    A1=L*u**nu*hyp2f1(-nu/2,0.5,1.5,-(L/u)**2); A2=((L*L+u*u)**(nu/2+1)-u**(nu+2))/(nu+2)
    B1=L**(nu+1)/(nu+1); B2=L**(nu+2)/(nu+2)
    return L*A1-A2-(L*B1-B2)
def J_log(L,u):
    """int_0^L (L-s) * 0.5*ln((s^2+u^2)/s^2) ds, elementary (logarithmic case alpha_s = 2)."""
    # int_0^L ln(s^2+u^2) ds = L ln(L^2+u^2) - 2L + 2u arctan(L/u) ; int_0^L ln s^2 ds = 2(L ln L - L)
    # int_0^L s ln(s^2+u^2) ds = 0.5[(L^2+u^2) ln(L^2+u^2) - (L^2+u^2) - u^2 ln u^2 + u^2] ; int_0^L s ln s^2 ds = L^2 ln L - L^2/2
    A=L*np.log(L*L+u*u)-2*L+2*u*np.arctan(L/u) - 2*(L*np.log(L)-L)
    B=0.5*((L*L+u*u)*np.log(L*L+u*u)-(L*L+u*u)-u*u*np.log(u*u)+u*u) - (L*L*np.log(L)-L*L/2)
    return 0.5*(L*A-B)
# Euler constant; upper spatial frequency of the mid band 1/(0.1 mm) for the logarithmic case
GAMMA_E=0.5772156649015329; F_MAX_LOG=1/0.1e-3
def D_psi(u,k,dx,D_m,alpha,phi2,f_max=F_MAX_LOG):
    """phi2 = (2 k0 sigma_mid)^2; returns D_psi(u) after k round trips with step dx. alpha_s <= 2.05: logarithmic
    structure function D_h(r) = (2/ln(f2 D_m)) [ln(pi f2 r) + gamma_E]."""
    L=k*dx
    if alpha<=2.05:
        return phi2*k*k*(2/L**2)*(2/np.log(f_max*D_m))*J_log(L,u)
    nu=alpha-2; return phi2*k*k*(2/L**2)*C_alpha(alpha)*D_m**(-nu)*J(L,u,nu)
def n_eff_closed(k,dx,D_m,alpha,sig_mid,sig_fein,c,wl,f_max=F_MAX_LOG,D_fig_c=0.0):
    """n_eff = 1/A^2 in closed form. Chord mean with a local power law D_psi ~ 2 (u/ell)^mu whose exponent is evaluated at
    the coherence length ell (D_psi(ell) = 2): start value mu from u = c/2 .. c, then one step at ell (explicit, no
    iteration to convergence). Against the quadrature of the chord integral: median 1.01, P90 1.08 for
    alpha_s = 2 ... 3.5 (R54, step 3). Returns (n_eff, mu, D_psi(c)); (1, 2, 0) if there is no phase disturbance."""
    k0=2*np.pi/wl; phi2=(2*k0*sig_mid)**2
    # D_fig_c (R63): structure function of the figure at the chord. Across the dispersion the figure is mainly a tilt; two strips at
    # distance u are shifted against each other after k round trips by k sqrt(12) sigma_perp,fig u/c: D_psi,fig(u) = D_fig_c (u/c)^2,
    # D_fig_c = 12 k^2 sigma_perp,fig^2. Alone it gives A^2 = sqrt(pi/a) erf(sqrt a) - (1 - e^-a)/a with a = D_fig_c/2 (same expression, mu = 2).
    D=lambda u: (D_psi(u,k,dx,D_m,alpha,phi2,f_max) if phi2>0 else 0.0)+D_fig_c*(u/c)**2
    Dc=D(c)
    if not np.isfinite(Dc) or Dc<=0: return 1.0, 2.0, 0.0
    # local exponent from D(c)/D(c/2), coherence length l from D_psi(l) = 2; then one refinement step at l (clipped to the chord)
    mu=float(np.clip(np.log(Dc/D(c/2))/np.log(2),0.2,2.5)); l=c*(2/Dc)**(1/mu)
    lu=min(l,c); mu=float(np.clip(np.log(D(lu)/D(lu/2))/np.log(2),0.2,2.5)); l=lu*(2/D(lu))**(1/mu)
    # chord integral via incomplete gamma functions; the fine band (independent strips) multiplies A^2 by exp(-(2 k0 sigma_fein)^2 k)
    z=(c/l)**mu; g1=gamma(1/mu)*gammainc(1/mu,z); g2=gamma(2/mu)*gammainc(2/mu,z)
    A2=(2/(mu*c*c))*(c*l*g1-l*l*g2)*np.exp(-(2*k0*sig_fein)**2*k)
    return 1.0/max(A2,1e-9), mu, Dc
