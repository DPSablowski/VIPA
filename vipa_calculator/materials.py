"""Refractive index and group index of some gap media (Sellmeier, lambda in um). Only a suggestion for the input."""
import math

SELLMEIER = {
    # Malitson 1965
    "Fused silica": ((0.6961663, 0.4079426, 0.8974794), (0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2)),
    # Schott data sheet
    "N-BK7": ((1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)),
    # Malitson 1963
    "CaF2": ((0.5675888, 0.4710914, 3.8484723), (0.050263605 ** 2, 0.1003909 ** 2, 34.649040 ** 2)),
}
MATERIALS = ["Air"] + list(SELLMEIER)


def n_sellmeier(name, wl_um):
    B, C = SELLMEIER[name]; l2 = wl_um * wl_um
    return math.sqrt(1 + sum(b * l2 / (l2 - c) for b, c in zip(B, C)))


def index(name, wl_nm):
    """(n, n_g) at wl_nm; air: (1, 1)."""
    if name == "Air": return 1.0, 1.0
    wl = wl_nm * 1e-3; d = 1e-4
    n = n_sellmeier(name, wl); dn = (n_sellmeier(name, wl + d) - n_sellmeier(name, wl - d)) / (2 * d)
    return n, n - wl * dn
