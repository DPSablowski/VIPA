"""Measures of one order from a spectrum I(delta) over the full period (FFT grid, NF points).

Computes the envelope fit (hull, flank fit), equivalent width, half-maximum crossing, 50 % energy width and the
overlap finesse F_om (Omega0 = 50/25/10 %), all as finesse 2 pi / width.

Library module (no command-line use, no files). Main entry: measures(I) returns the tuple in the order of NAMES;
measures_hm(I) only the half-maximum value; overlap_F(I) the overlap finesse values.
"""
import numpy as np
import pupil_theory_v7 as PT, run_full_grid_v7 as G
# FFT grid of the theory spectra (NF points over one period), sorted phase axis DS;
# overlap measures on OV_N bins with the thresholds 50 / 25 / 10 %
NF = 2 ** 14; D0 = np.fft.fftfreq(NF) * 2 * np.pi; ORD = np.argsort(D0); DS = D0[ORD]
OV_N = 8192; OV_THR = (0.5, 0.25, 0.10)

def omega_cross(p, thresholds=OV_THR):
    """Omega(s) = sum min(p, p shifted by s) for an energy profile p over the period. s_Omega = LAST crossing of Omega
    with the threshold (up to half the period); coarse grid (steps of N/512), then the last crossing refined densely and
    interpolated linearly (no rounding to the grid). Returns the list of finesse values N / s, one per threshold."""
    N = len(p); half = N // 2; cs = max(1, N // 512); cache = {}
    def om(k):
        k = int(k)
        if k not in cache: cache[k] = float(np.minimum(p, np.roll(p, k)).sum())
        return cache[k]
    coarse = np.arange(1, half + 1, cs); omc = np.array([om(k) for k in coarse]); out = []
    for Om in thresholds:
        bad = np.where(omc > Om)[0]
        if len(bad) == 0: out.append(N / 1.0); continue
        kc = int(coarse[bad[-1]])
        if kc + cs >= half and om(half) > Om: out.append(N / half); continue         # not separable up to half the period
        ks = np.arange(kc, min(kc + cs, half) + 1); oms = np.array([om(k) for k in ks]); b2 = np.where(oms > Om)[0]; kl = int(ks[b2[-1]])
        if kl >= half: out.append(N / half); continue
        o1, o2 = om(kl), om(kl + 1); s_ = kl + (o1 - Om) / (o1 - o2) if o1 > o2 else kl + 1.0
        out.append(N / s_)                                                           # F = 2 pi / (s * 2 pi / N)
    return out

def overlap_F(I):
    """I: intensity on the FFT grid (one period); binned to OV_N points (energy conserved), normalised to energy 1.
    Returns the overlap finesse values for the thresholds OV_THR."""
    p = np.asarray(I, float).reshape(OV_N, -1).sum(1); return omega_cross(p / p.sum())

def energy_measures(d, I):
    """Returns (equivalent-width finesse 2 pi / (energy / peak), 50 % energy-width finesse 2 pi / W_50), where W_50 is
    the width between the 25 % and 75 % points of the cumulative energy of the line centred at its maximum."""
    dd = d[1] - d[0]; tot = I.sum() * dd; W_eq = tot / I.max(); i = int(np.argmax(I)); Ir = np.roll(I, len(I) // 2 - i); c = np.cumsum(Ir) / Ir.sum()
    W_50 = (np.searchsorted(c, 0.75) - np.searchsorted(c, 0.25)) * dd
    return 2 * np.pi / W_eq, (2 * np.pi / W_50 if W_50 > 0 else np.nan)

# order of the measures returned by measures()
NAMES = ("hull", "eq", "hm", "e50", "om50", "om25", "om10")

def hm_zentriert(I):
    """Outer half-maximum crossing as finesse 2 pi / width. The line is centred first (grid rolled around the maximum,
    positions relative to the maximum): on the uncentred grid (-pi ... pi) a line extending over the edge would count
    with almost 2 pi width (relevant for a strong wedge)."""
    ip = int(np.argmax(I)); n = len(I); J = np.roll(I, n // 2 - ip); x = (np.arange(n) - n // 2) * 2 * np.pi / n
    w = G.halfmax_fwhm(x, J / J.max()); return 2 * np.pi / w if w else np.nan
def measures(I):
    """I: intensity on the FFT grid (unsorted, np.fft.fftfreq order). Returns a tuple in the order of NAMES."""
    Is = I[ORD]; F1, F2 = PT.measure(DS, Is)
    Fhm = hm_zentriert(I)
    _, Fe50 = energy_measures(DS, Is)
    return (F1, F2, Fhm, Fe50) + tuple(overlap_F(Is))


def measures_hm(I):
    """Only the half-maximum finesse, as a tuple in the order of NAMES (the others NaN)."""
    out = [np.nan] * len(NAMES); out[NAMES.index("hm")] = hm_zentriert(I); return tuple(out)
