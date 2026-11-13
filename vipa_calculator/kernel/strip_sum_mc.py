"""Strip sum with n_eff independent strips (calculation R55 step 6 of the main document).

Idea
    The incoherent sum over the chord lies between the line of a single strip (concentrated) and the ensemble
    line (smooth). What matters is the number of independent strips n_eff = 1/A_rms^2(k*), with A_rms^2 the power
    of the coherent chord sum (apod.py) at round trip k* = 2/D0 (coherence length of the line). The line of the
    sum is simulated in 1D with the residual spectrum of a strip: n_eff independent step sequences, for each strip
    a line with decay, edge and wedge, added incoherently; the median over realisations is returned.

Contents (library module, no main program)
    strip_spectrum  -- phase-step power spectrum of one strip from the three surface bands
    F_sum_mc        -- Monte-Carlo finesse (overlap measure) of the incoherent strip sum
    n_eff_of        -- number of independent strips from the A_rms profile

Inputs
    A RestSpektrum object (res_eff.py) with the residual spectra of the bands; messung.py (overlap_F) and apod.py.

"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import numpy as np
import messung as MS
from res_eff import RestSpektrum
from apod import A_rms_profile

def strip_spectrum(RS, sig_fig, sig_mid, sig_fein):
    """Returns (S, NS): the one-sided power spectrum S of the phase steps per round trip of one strip on the
    rfft grid of length NS = 4 M, summed from the residual spectra of figure, mid band and fine band
    (rms heights sig_fig, sig_mid, sig_fein), and the grid length NS."""
    # c2 = (2 k0)^2 converts surface height to reflection phase; the discrete lines of mid band and figure are
    # binned onto the frequency grid (searchsorted), the fine band is white (spread evenly over all bins)
    c2 = (2*RS.k0)**2; NS = 4*RS.M; om = 2*np.pi*np.fft.rfftfreq(NS); S = np.zeros(len(om))
    if sig_mid > 0: idx = np.clip(np.searchsorted(om, np.abs(RS.mid_om)), 0, len(om) - 1); np.add.at(S, idx, c2*sig_mid**2*RS.mid_rest)
    if sig_fig > 0: idx = np.clip(np.searchsorted(om, np.abs(RS.om_fft)), 0, len(om) - 1); np.add.at(S, idx, c2*sig_fig**2*RS.fig)
    if sig_fein > 0: S += c2*sig_fein**2*RS.fein_rest/len(om)
    return S, NS

def F_sum_mc(RS, sig_fig, sig_mid, sig_fein, gam, beta, n_eff, sperp2=0.0, src_pm=0.0, FSR_pm=None, n_real=24, NF=2**14, seed=2, var_q=0.0):
    """Returns the median over n_real realisations of the overlap-measure finesse (messung.overlap_F) of the
    incoherent sum of round(n_eff) independent strips. Each strip: window exp(-gam m), chirp bq m^2 and a random
    phase walk drawn from strip_spectrum. Optional Gaussian broadening by the strip offset (variance sperp2, rad^2)
    and by a Gaussian source of FWHM src_pm (needs FSR_pm); var_q = variance of the random chirp of the plate."""
    rng = np.random.default_rng(seed); M = RS.M; m = np.arange(M)
    S, NS = strip_spectrum(RS, sig_fig, sig_mid, sig_fein); tot = S.sum(); amp = np.sqrt(S)
    n = max(1, int(round(n_eff))); Fs = []
    kk = np.fft.fftfreq(NF)*NF
    for _ in range(n_real):
        # random chirp of the plate (common to all strips)
        bq = beta + 0.5*np.sqrt(var_q)*rng.standard_normal()
        P0 = np.exp(-gam*m + 1j*bq*m**2); I = np.zeros(NF)
        for _s in range(n):
            # Gaussian random phase steps with spectrum S (random complex amplitudes), rescaled to the exact total
            # power; their cumulative sum is the phase walk along the round trips
            z = np.fft.irfft(amp*(rng.standard_normal(len(amp)) + 1j*rng.standard_normal(len(amp))), NS)[:M]
            if tot > 0 and np.mean(z**2) > 0: z = z*np.sqrt(tot/np.mean(z**2))
            I += np.abs(np.fft.fft(P0*np.exp(1j*np.cumsum(z)), NF))**2
        if sperp2 > 0 or (src_pm > 0 and FSR_pm):
            # Gaussian broadening: multiply the autocorrelation (FFT of I) by exp(-sg2 k^2/2);
            # source sigma in rad = 2 pi pm/FSR / 2.3548 (FWHM -> sigma)
            sg2 = sperp2 + ((2*np.pi*src_pm/FSR_pm/2.3548)**2 if (src_pm > 0 and FSR_pm) else 0.0)
            I = np.real(np.fft.ifft(np.fft.fft(I)*np.exp(-0.5*sg2*kk**2)))
        Fs.append(MS.overlap_F(np.maximum(I, 0))[0])
    return float(np.median(Fs))

def n_eff_of(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, sig_fig, sig_mid, alpha_s, sig_fein, D0):
    """Returns (n_eff, A): the number of independent strips n_eff = 1/A_rms^2(k*) at k* = round(2/D0)
    (A^2 floored at 1e-6) and the full A_rms profile over the round-trip lag from apod.A_rms_profile."""
    A, w = A_rms_profile(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, sig_fig, sig_mid, alpha_s, sig_fein, nf=40, nphi=24)
    k = min(int(round(2.0/D0)), len(A) - 1); return 1.0/max(A[k]**2, 1e-6), A
