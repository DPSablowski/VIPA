"""closed_model3.py -- coherent ray model of a VIPA with a Zernike HR surface (direct ray summation).

Purpose:
    Rays start in the focal spot of the multimode core image at the edge of the plate (x = -R_mirror), spread over the
    beam width in y and over the divergence angle in x. Each round trip the ray is reflected at the HR surface,
    which carries a random Zernike figure error (terms 2..36 without piston, tilt, defocus, astigmatism, see
    zernike_hr_surface); the height adds the phase 2 k0 h, the slope tilts the ray by 2 dh/dx, 2 dh/dy, and the ray
    advances by the round-trip offset Delta_x plus angle * L_step. Rays leaving the plate (|x| > R_mirror) are dropped.
    The fields of all round trips m are summed coherently with the phase m*delta.
    Speed-ups: the Zernike basis and its gradients dZ/dx, dZ/dy are evaluated analytically (chain rule in polar
    coordinates) for all active terms at once.
Functions:
    basis_and_grad   -- Zernike basis and gradients at given points;
    coherent_F       -- median finesse over n_real surface realizations (Lorentz fit to both flanks);
    coherent_spectrum-- profile (delta, I) of one realization over a full period, for the measurement with the
                        pipeline's own width functions.
Fixed geometry (module constants): plate radius 12.5 mm, cylinder lens focal length 75 mm, theta = 4 deg,
round-trip offset Delta_x, path step L_step, k0 = 2 pi / 630 nm, core image 75 um.
Environment variable VIPA_MEAN_VAR_LEGACY=1 selects the legacy normalization of the surface variance (see
_calibrate_mean_var); default is the exact one.
Library module, no main program; it imports zernike_hr_surface from the current directory.

Used in the main document: appendix H.7 (Produktionsraster, gemeinsame Zufallszahlen und Nachrechnung). Pages: 353.
"""
import numpy as np, sys
sys.path.insert(0,'.')
from scipy.special import factorial
from zernike_hr_surface import noll_to_nm

# Zernike terms left out: piston, tilt, defocus, astigmatism; the remaining terms up to Noll index 36 are active
EXCLUDED = {(0,0),(1,1),(1,-1),(2,0),(2,2),(2,-2)}
_active = [noll_to_nm(j) for j in range(1,37) if noll_to_nm(j) not in EXCLUDED]
NACT=len(_active)

# geometry of the standard VIPA in metres / radians; theta = 4 deg; k0 = 2 pi / 630 nm
R_mirror=12.5e-3; f_cyl=0.075; theta=0.06981317007977318
Delta_x=0.00027970724777404165; L_step=0.0040097675923246884; k0=9973310.011396168
d_core_image=75e-6

def _radial_coeffs(n,m):
    """Return the list of (coefficient, power of rho) of the Zernike radial polynomial R_n^|m|."""
    m=abs(m); cs=[]
    for k in range((n-m)//2+1):
        c=(-1)**k*factorial(n-k)/(factorial(k)*factorial((n+m)//2-k)*factorial((n-m)//2-k))
        cs.append((c,n-2*k))
    return cs
_rad = [_radial_coeffs(n,m) for (n,m) in _active]
_norm = [np.sqrt(n+1) if m==0 else np.sqrt(2*(n+1)) for (n,m) in _active]

def basis_and_grad(X,Y):
    """X, Y: (N,) arrays of positions [m]. Returns Z (NACT,N), dZdx (NACT,N), dZdy (NACT,N); zero outside the disk."""
    rho=np.sqrt(X**2+Y**2)/R_mirror; phi=np.arctan2(Y,X)
    inside=(rho<=1.0)
    rho_s=np.where(inside,rho,0.0)  # avoids the warning for 0 to a negative power; masked later
    N=len(X)
    Z=np.zeros((NACT,N)); dZdrho=np.zeros((NACT,N)); Zang=np.zeros((NACT,N)); dZdphi=np.zeros((NACT,N))
    for i,((n,m),cs,norm) in enumerate(zip(_active,_rad,_norm)):
        Rp=np.zeros(N); dRp=np.zeros(N)
        for c,p in cs:
            Rp += c*rho_s**p
            dRp += c*p*rho_s**(p-1) if p>0 else 0.0
        if m==0: ang=np.ones(N); dang=np.zeros(N)
        elif m>0: ang=np.cos(m*phi); dang=-m*np.sin(m*phi)
        else: ang=np.sin(abs(m)*phi); dang=abs(m)*np.cos(abs(m)*phi)
        Z[i]=norm*Rp*ang; dZdrho[i]=norm*dRp*ang; dZdphi[i]=norm*Rp*dang
    # chain rule: rho = sqrt(x^2+y^2)/Rm, phi = atan2(y,x); 1e-15 avoids division by zero at the centre
    r_cart=np.sqrt(X**2+Y**2)+1e-15
    drho_dx=X/(r_cart*R_mirror); drho_dy=Y/(r_cart*R_mirror)
    dphi_dx=-Y/(r_cart**2); dphi_dy=X/(r_cart**2)
    dZdx = dZdrho*drho_dx[None,:] + dZdphi*dphi_dx[None,:]
    dZdy = dZdrho*drho_dy[None,:] + dZdphi*dphi_dy[None,:]
    Z[:,~inside]=0.0; dZdx[:,~inside]=0.0; dZdy[:,~inside]=0.0
    return Z,dZdx,dZdy

def _calibrate_mean_var(legacy=False):
    """Return the mean surface variance per unit coefficient: mean over the WHOLE disk of sum_j Z_j^2
    (orthonormal modes: exactly NACT). It is used to scale the random coefficients to a surface RMS sw.
    legacy=True: approximate calibration on a square grid |x|,|y| <= 0.95 without the edge ring (value 28.438
    instead of 30; surfaces then have an RMS 2.6 % too large, 1.026 sigma_w). Needed only to reproduce archived
    runs; selected with the environment variable VIPA_MEAN_VAR_LEGACY=1."""
    if legacy:
        ref=np.linspace(-0.95,0.95,40); Xr,Yr=np.meshgrid(ref,ref); Xr,Yr=Xr.ravel(),Yr.ravel()
        m=Xr**2+Yr**2<=1; Xr,Yr=Xr[m]*R_mirror,Yr[m]*R_mirror
        Z,_,_=basis_and_grad(Xr,Yr)
        return np.mean(np.sum(Z**2,axis=0))
    # polar quadrature over the disk: Gauss-Legendre in u = rho^2 (equal area weights), equidistant in phi
    xu,wu=np.polynomial.legendre.leggauss(96); u=0.5*(xu+1.0); wu=0.5*wu; ph=np.linspace(0,2*np.pi,128,endpoint=False)
    U_,P_=np.meshgrid(u,ph); rr=np.sqrt(U_)*R_mirror; Xq=(rr*np.cos(P_)).ravel(); Yq=(rr*np.sin(P_)).ravel(); W=np.tile(wu,(len(ph),1)).ravel()/len(ph)
    Z,_,_=basis_and_grad(Xq,Yq)
    return float(np.sum(W*np.sum(Z**2,axis=0)))
import os as _os
MEAN_VAR=_calibrate_mean_var(legacy=_os.environ.get("VIPA_MEAN_VAR_LEGACY","0")=="1")

def coherent_F(R2, sw, bl, pmax, n_sub=15, n_real=20, seed0=0, window_mult=2.0, n_scan=4000, use_yspread=True, use_xangle=True):
    """Return (median finesse, number of valid realizations) for reflectivity R2, surface RMS sw [m], beam width bl [mm]
    and pmax round trips. Rays: n_sub x n_sub grid inside a circle (equal weights). Each realization: coherent sum
    over round trips on the window |delta| <= 0.4 pi * window_mult, then a Lorentz fit to the left and right flank
    (points between 5 % and 60 % of the maximum); F = 2 pi / (gamma_left + gamma_right).
    use_yspread / use_xangle switch the y spread of the start points and the x divergence on or off."""
    r2=np.sqrt(R2); d_blende=bl*1e-3; theta_div=np.arctan(d_blende/(2*f_cyl))
    xi0=np.linspace(-1,1,n_sub); XI_X,XI_Y=np.meshgrid(xi0,xi0)
    circ=(XI_X**2+XI_Y**2)<=1.0; w_full=circ.astype(float).ravel(); active_rays=w_full>0
    weights=w_full[active_rays]; weights/=weights.sum()
    x0=(-R_mirror+d_core_image/2+XI_X*(d_core_image/2)).ravel()[active_rays]  # start spot at the plate edge, as in the ray tracer
    y0=(XI_Y*(d_blende/2)).ravel()[active_rays] if use_yspread else np.zeros(active_rays.sum())
    thx0=(XI_X*theta_div).ravel()[active_rays] if use_xangle else np.zeros(active_rays.sum())
    n_rays=len(weights)
    window=0.4*np.pi*window_mult; delta=np.linspace(-window,window,n_scan)
    m_arr=np.arange(pmax+1)
    rng=np.random.default_rng(seed0)
    Fs=[]
    for real in range(n_real):
        # random Zernike coefficients with expected surface RMS sw
        coeffs=rng.standard_normal(NACT)*np.sqrt(sw**2/MEAN_VAR)
        x=x0.copy(); y=y0.copy(); thx=thx0.copy(); thy=np.zeros_like(thx)
        alive=np.ones(n_rays,dtype=bool); acc_phase=np.zeros(n_rays)
        Ereal=np.zeros(n_scan); Eimag=np.zeros(n_scan)
        # propagate all rays round trip by round trip and add their fields
        for m in m_arr:
            alive = alive & (np.abs(x)<=R_mirror)
            Zb,dZx,dZy = basis_and_grad(x,y)
            hgt = coeffs@Zb; hx = coeffs@dZx; hy = coeffs@dZy
            # reflection phase 2 k0 h; amplitude r^m (r = sqrt(R2)), zero once the ray has left the plate
            acc_phase = acc_phase + k0*2*hgt
            amp = weights*r2**m*alive
            phase_per_ray = acc_phase[:,None] + m*delta[None,:]
            Ereal += (amp[:,None]*np.cos(phase_per_ray)).sum(axis=0)
            Eimag += (amp[:,None]*np.sin(phase_per_ray)).sum(axis=0)
            # a surface slope tilts the reflected ray by twice the slope; then step to the next round trip
            thx = thx + 2*hx; thy = thy + 2*hy
            x = x + Delta_x + thx*L_step; y = y + thy*L_step
        I = Ereal**2+Eimag**2
        # select the flank points (between 5 % and 60 % of the maximum) on each side for the Lorentz fit
        Imax=I.max(); i0=np.argmax(I); frac_lo,frac_hi=0.05,0.6
        i=0
        while i<i0 and I[i]<frac_hi*Imax: i+=1
        left=np.arange(0,i+1); left=left[I[left]>=frac_lo*Imax]
        j=n_scan-1
        while j>i0 and I[j]<frac_hi*Imax: j-=1
        right=np.arange(j,n_scan); right=right[I[right]>=frac_lo*Imax]
        if len(left)<2 or len(right)<2: continue
        from scipy.optimize import curve_fit
        x0d=delta[i0]
        def lor(xx,gam): return Imax/(1+((xx-x0d)/gam)**2)
        try:
            pl,_=curve_fit(lor,delta[left],I[left],p0=[window/10],bounds=(1e-6,window))
            pr,_=curve_fit(lor,delta[right],I[right],p0=[window/10],bounds=(1e-6,window))
            Fs.append(2*np.pi/(pl[0]+pr[0]))
        except Exception: continue
    return (np.median(Fs) if Fs else np.nan), len(Fs)


def coherent_spectrum(R2, sw, bl, pmax, n_sub=15, seed=0, n_scan=6000, use_yspread=True, use_xangle=True):
    """Like coherent_F, but returns (delta, I) of ONE realization over a full period [-pi, pi], so that the width
    can be measured with the pipeline's own width functions. The sum over rays is done per round trip first
    (A, B = real and imaginary part of the summed field), then multiplied by exp(i m delta)."""
    r2=np.sqrt(R2); d_blende=bl*1e-3; theta_div=np.arctan(d_blende/(2*f_cyl))
    xi0=np.linspace(-1,1,n_sub); XI_X,XI_Y=np.meshgrid(xi0,xi0)
    circ=(XI_X**2+XI_Y**2)<=1.0; w_full=circ.astype(float).ravel(); active_rays=w_full>0
    weights=w_full[active_rays]; weights/=weights.sum()
    x0=(-R_mirror+d_core_image/2+XI_X*(d_core_image/2)).ravel()[active_rays]
    y0=(XI_Y*(d_blende/2)).ravel()[active_rays] if use_yspread else np.zeros(active_rays.sum())
    thx0=(XI_X*theta_div).ravel()[active_rays] if use_xangle else np.zeros(active_rays.sum())
    n_rays=len(weights); delta=np.linspace(-np.pi,np.pi,n_scan)
    rng=np.random.default_rng(seed)
    coeffs=rng.standard_normal(NACT)*np.sqrt(sw**2/MEAN_VAR)
    x=x0.copy(); y=y0.copy(); thx=thx0.copy(); thy=np.zeros_like(thx)
    alive=np.ones(n_rays,dtype=bool); acc=np.zeros(n_rays)
    Er=np.zeros(n_scan); Ei=np.zeros(n_scan)
    for m in range(pmax+1):
        alive=alive&(np.abs(x)<=R_mirror)
        Zb,dZx,dZy=basis_and_grad(x,y)
        acc=acc+k0*2*(coeffs@Zb); hx=coeffs@dZx; hy=coeffs@dZy
        amp=weights*r2**m*alive
        A=np.sum(amp*np.cos(acc)); B=np.sum(amp*np.sin(acc))
        cg=np.cos(m*delta); sg=np.sin(m*delta)
        Er+=A*cg-B*sg; Ei+=A*sg+B*cg
        thx=thx+2*hx; thy=thy+2*hy
        x=x+Delta_x+thx*L_step; y=y+thy*L_step
    return delta, Er**2+Ei**2
