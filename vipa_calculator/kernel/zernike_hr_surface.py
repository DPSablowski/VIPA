"""Random HR-mirror surface built from Zernike polynomials.

Purpose
    Provides normalised Zernike polynomials (Noll normalisation: unit RMS over the unit disc), the conversion from
    the Noll index j to (n, m), and build_zernike_surface, which draws a random surface from the Zernike terms
    j = 2..n_terms without piston, tilt, defocus and astigmatism, scaled to a prescribed RMS over the circular aperture.
    The functions zernike / zernike_radial are also used by the three-band surface model (figure band).

Inputs / outputs
    Library module. When run directly it builds one example surface (630 nm, RMS = lambda/10, radius 12.5 mm), prints
    the number of terms, RMS and PV, and saves a plot to /mnt/user-data/outputs/zernike_hr_surface.png (path of the
    original runs; adapt as needed).

Run
    python3 zernike_hr_surface.py

Used in the main document: chapter 5 (Strahlverfolgung und Prüfung der Theorie); appendix O.3 (Skripte). Pages: 39,
443.
"""
import numpy as np
from scipy.special import factorial

def zernike_radial(n, m, rho):
    """Return the radial Zernike polynomial R_n^|m|(rho) (explicit factorial sum)."""
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n-m)//2 + 1):
        c = (-1)**k * factorial(n-k) / (factorial(k)*factorial((n+m)//2-k)*factorial((n-m)//2-k))
        R += c * rho**(n-2*k)
    return R

def zernike(n, m, rho, phi):
    """Return the Noll-normalised Zernike polynomial Z_n^m(rho, phi): cos(m phi) for m > 0, sin(|m| phi) for m < 0,
    normalisation sqrt(n+1) (m = 0) or sqrt(2(n+1)) (m != 0), so that each term has unit RMS on the unit disc."""
    R = zernike_radial(n, abs(m), rho)
    if m == 0:
        return np.sqrt(n+1) * R
    elif m > 0:
        return np.sqrt(2*(n+1)) * R * np.cos(m*phi)
    else:
        return np.sqrt(2*(n+1)) * R * np.sin(abs(m)*phi)

def noll_to_nm(j):
    """Return (n, m) for the Noll index j (j = 1 is piston); the sign of m follows the parity of j."""
    n = 0
    j1 = j-1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2))/2))
    return n, m

# Excluded terms: piston, tilt, defocus, astigmatism (these are compensated by alignment/focus or have no effect)
EXCLUDED = {(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2)}

def build_zernike_surface(R_mirror, sigma_target, rng, n_terms=36, grid_n=801, power_law_p=0):
    """Build a 2D surface from randomly weighted Zernike terms 2..n_terms (without piston/tilt/defocus/astigmatism),
    normalised to RMS = sigma_target over the circular aperture.

    R_mirror: aperture radius [m]; rng: numpy Generator; grid_n: grid points per axis.
    power_law_p: each coefficient is weighted with 1/n^power_law_p before drawing (p = 0: all orders weighted
    equally; p > 0: low orders dominate the error budget, more realistic for real polishing).
    Returns (grid coordinates xg, surface heights surf (zero outside the aperture), aperture mask, list of (n, m))."""
    xg = np.linspace(-R_mirror, R_mirror, grid_n)
    X, Y = np.meshgrid(xg, xg)
    rho = np.sqrt(X**2+Y**2)/R_mirror
    phi = np.arctan2(Y, X)
    mask = rho <= 1.0

    active_terms = []
    for j in range(1, n_terms+1):
        n,m = noll_to_nm(j)
        if (n,m) in EXCLUDED:
            continue
        active_terms.append((n,m))

    coeffs = rng.normal(0,1,size=len(active_terms))
    if power_law_p != 0:
        weight_n = np.array([1.0/(n**power_law_p) for (n,m) in active_terms])
        coeffs = coeffs*weight_n

    surf = np.zeros_like(rho)
    for c,(n,m) in zip(coeffs, active_terms):
        Z = zernike(n,m,rho,phi)
        Z[~mask] = 0
        surf += c*Z

    # Normalise to the requested RMS (inside the aperture only)
    current_rms = np.std(surf[mask])
    surf *= sigma_target/current_rms
    surf[~mask] = 0
    return xg, surf, mask, active_terms

if __name__=="__main__":
    rng = np.random.default_rng(42)
    wavelength = 630e-9
    sigma_target = wavelength/10
    # Radius; plate extent ~25 mm diameter (L_eff)
    R_mirror = 12.5e-3

    xg, surf, mask, terms = build_zernike_surface(R_mirror, sigma_target, rng)
    print(f'Anzahl verwendeter Zernike-Terme (ohne Piston/Tilt/Defocus): {len(terms)}')
    print(f'Ziel-RMS: {sigma_target*1e9:.2f}nm, Erreichte RMS: {np.std(surf[mask])*1e9:.2f}nm')
    print(f'PV (peak-valley): {(surf[mask].max()-surf[mask].min())*1e9:.2f}nm')

    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6,5))
    surf_show = np.where(mask, surf*1e9, np.nan)
    im = ax.imshow(surf_show, extent=[-R_mirror*1e3,R_mirror*1e3,-R_mirror*1e3,R_mirror*1e3], cmap='RdBu_r')
    plt.colorbar(im, label='Hoehe [nm]')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title(f'HR-Spiegel Oberflaeche (Zernike 2-36, ohne Tilt/Defocus, RMS={sigma_target*1e9:.1f}nm)')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/zernike_hr_surface.png', dpi=140)
    print('saved')
