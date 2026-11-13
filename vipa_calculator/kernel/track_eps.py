"""Shift of the hits caused by the beam deflection and the resulting phase
error in the three-band model.

Physics
    Mid band: slope covariance C_s(k) = (2 pi)^2 sigma^2 sum dens fx^2 cos(2 pi fx k dx) (isotropic power law,
    band 1/D_m .. 1/0.1 mm);
    figure: slope constant over the window (Zernike gradient); the fine band does not contribute (white: same
    statistics at every position).
    Var dx_m = 16 h^2 sum_{i,k<m} (m-i)(m-k) C_s(|i-k|);  phase error eps^2 = (2 k0)^2 [D_mid(dx) + sigma_s,fig^2 dx^2],
    with the structure function D_mid of the mid band.

Functions
    fig_slope_factor()  mean squared x-gradient of the figure Zernike terms on the unit disc (stored in SF)
    spec(alpha_s, D_m)  cached frequency grid and normalised power-law weights
    eps_track(...)      shift and phase error for one geometry

Run
    python3 track_eps.py   (prints the slope factor and four example geometries)

"""
import sys; pass; import numpy as np
from flaeche_dreiband import FIG_TERMS, FMAX_DEFAULT
from zernike_hr_surface import zernike
def fig_slope_factor():
    """Return <(dZ/dx)^2>, averaged over the figure Zernike terms and the unit disc (edge rho > 0.98 excluded,
    where the numerical gradient is distorted).
    """
    n=401; x=np.linspace(-1,1,n); X,Y=np.meshgrid(x,x); rho=np.sqrt(X**2+Y**2); phi=np.arctan2(Y,X); ins=rho<=0.98; d=x[1]-x[0]; tot=0.0
    for nn,mm_ in FIG_TERMS:
        Z=zernike(nn,mm_,np.minimum(rho,1),phi)*(rho<=1); gx=np.gradient(Z,d,axis=1); tot+=np.mean(gx[ins]**2)
    # <(dZ/dx)^2> averaged over the terms and the disc (unit radius)
    return tot/len(FIG_TERMS)
SF=fig_slope_factor()
# cache of frequency grids per (alpha_s, D_m)
_sp={}
def spec(alpha_s,D_m,nf=160,nphi=48):
    """Return (fx, dens): x components of a polar frequency grid from 1/D_m to f_max and the power-law weights
    f^(2-alpha_s), normalised to sum 1 (cached).
    """
    key=(round(alpha_s,3),round(D_m,5))
    if key not in _sp:
        f=np.geomspace(1/D_m,FMAX_DEFAULT,nf); ph=(np.arange(nphi)+0.5)*2*np.pi/nphi; F,PH=np.meshgrid(f,ph,indexing="ij"); fx=(F*np.cos(PH)).ravel()
        dens=(F**(2-alpha_s)).ravel(); dens/=dens.sum(); _sp[key]=(fx,dens)
    return _sp[key]
def eps_track(R2,M_k,theta_deg,wl_nm,h,D_m,alpha_s,sig_mid,sig_fig):
    """Shift of the hits and phase error; h, D_m and sig in m.

    Returns (N_w, dx_tot [m], eps [rad], fraction of the figure in the shift variance, sigma_s_mid, sigma_s_fig [rad]).
    """
    # gamma = -ln r; window of N_w ~ 1/gamma round trips (at most M_k); dx = round-trip step; weights w = N_w - k
    gam=-0.5*np.log(R2); Nw=int(max(2,min(M_k,round(1/gam)))); dx=2*h*np.tan(np.deg2rad(theta_deg)); k0=2*np.pi/(wl_nm*1e-9)
    fx,dens=spec(alpha_s,D_m); k=np.arange(Nw); w=(Nw-k).astype(float)
    # Toeplitz matrix of the slope covariance of the mid band
    Cs=np.array([(2*np.pi)**2*sig_mid**2*np.sum(dens*fx**2*np.cos(2*np.pi*fx*kk*dx)) for kk in range(Nw)])
    idx=np.abs(k[:,None]-k[None,:]); V_mid=16*h**2*float(w@Cs[idx]@w)
    # figure: rms slope (Zernike gradient scaled to the radius D_m/2), constant over the window
    s_s_fig=sig_fig*np.sqrt(SF)/(D_m/2); V_fig=16*h**2*s_s_fig**2*float(np.sum(w))**2
    dxt=np.sqrt(V_mid+V_fig); s_s_mid=np.sqrt(Cs[0])
    # structure function of the mid band at the total shift, and the phase error of the double pass (2 k0)
    D_mid=2*sig_mid**2*np.sum(dens*(1-np.cos(2*np.pi*fx*dxt)))
    eps=2*k0*np.sqrt(D_mid+s_s_fig**2*dxt**2)
    return Nw,dxt,eps,V_fig/(V_mid+V_fig+1e-300),s_s_mid,s_s_fig
if __name__=="__main__":
    print("Zernike-Neigungsfaktor <(dZ/dx)^2> (Einheitsscheibe):",round(SF,2))
    # examples: standard VIPA and stronger surfaces / higher reflectivity / larger mirror spacing
    for R2,M,th,wl,h,Dm,al,sm,sf in ((0.9,89,4,630,2e-3,0.0261,3.0,15e-9,10e-9),(0.9,89,4,630,2e-3,0.0261,2.0,50e-9,25e-9),(0.98,89,4,630,2e-3,0.0261,2.0,50e-9,25e-9),(0.98,40,6,630,8e-3,0.0261,2.0,50e-9,25e-9)):
        Nw,dxt,eps,fr,ss,sf_=eps_track(R2,M,th,wl,h,Dm,al,sm,sf); print(f"R2 {R2} h {h*1e3:.0f} sig_mid {sm*1e9:.0f} fig {sf*1e9:.0f} alpha_s {al}: N_w {Nw}, dx {dxt*1e6:.2f} um (Figur-Anteil {fr:.2f}), Neigung mid {ss*1e3:.2f} mrad fig {sf_*1e3:.3f} mrad, eps {eps:.3f} rad")
