"""Havlena-Odeh material balance engine (spec v3/v4, Sections 4.2-4.4).

Computes underground withdrawal (F), the expansion terms (Eo, Eg, Efw, Et),
the steady-state linear aquifer model, and the zero-intercept regression
used to fit every H-O straight line.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .pvt import interpolate_pvt_for_pressures

AQUIFER_CONSTANTS = {
    "None": 0,
    "Weak": 50_000,
    "Moderate": 200_000,
    "Strong": 800_000,
}


def underground_withdrawal(Np, Rs, Gp, Bo, Bg, Wp, Bw):
    """
    F = Np * Bo + (Gp - Np * Rs) * Bg + Wp * Bw
    Units: F in reservoir barrels (rb)

    (Gp - Np * Rs) is free gas produced and should be >= 0; a negative
    value means producing GOR is below solution GOR, which is flagged
    as a warning during validation, not enforced here.
    """
    return Np * Bo + (Gp - Np * Rs) * Bg + Wp * Bw


def oil_and_dissolved_gas_expansion(Bo, Boi, Rsi, Rs, Bg):
    """
    Eo = (Bo - Boi) + (Rsi - Rs) * Bg
    Units: rb/STB
    """
    return (Bo - Boi) + (Rsi - Rs) * Bg


def gas_cap_expansion(Boi, Bg, Bgi):
    """
    Eg = Boi * ((Bg / Bgi) - 1)
    Units: rb/STB
    """
    return Boi * ((Bg / Bgi) - 1)


def rock_and_water_expansion(Boi, cf, cw, Swc, Pi, P):
    """
    Efw = Boi * ((cf + cw * Swc) / (1 - Swc)) * (Pi - P)
    Units: rb/STB
    """
    return Boi * ((cf + cw * Swc) / (1 - Swc)) * (Pi - P)


def combined_expansion(Eo, Eg, Efw, m):
    """
    Et = Eo + m * Eg + Efw
    """
    return Eo + m * Eg + Efw


def compute_ho_terms(production_df, pvt_df, constants):
    """
    Computes the Havlena-Odeh terms at every production time step.

    Parameters:
        production_df : DataFrame with columns Np, Gp, Wp, P_avg
        pvt_df         : DataFrame with columns P, Rs, Bo, Bg, Bw
                          (PVT properties are interpolated onto each
                          production row's P_avg)
        constants      : dict with keys Pi, Boi, Bgi, Rsi, cf, cw, Swc, m

    Returns:
        DataFrame with columns [P, F, Eo, Eg, Efw, Et]
    """
    Pi = constants["Pi"]
    Boi = constants["Boi"]
    Bgi = constants["Bgi"]
    Rsi = constants["Rsi"]
    cf = constants["cf"]
    cw = constants["cw"]
    Swc = constants["Swc"]
    m = constants["m"]

    P = production_df["P_avg"].to_numpy(dtype=float)
    Np = production_df["Np"].to_numpy(dtype=float)
    Gp = production_df["Gp"].to_numpy(dtype=float)
    Wp = production_df["Wp"].to_numpy(dtype=float)

    pvt_at_P = interpolate_pvt_for_pressures(P, pvt_df)
    Rs = pvt_at_P["Rs"].to_numpy()
    Bo = pvt_at_P["Bo"].to_numpy()
    Bg = pvt_at_P["Bg"].to_numpy()
    Bw = pvt_at_P["Bw"].to_numpy()

    F = underground_withdrawal(Np, Rs, Gp, Bo, Bg, Wp, Bw)

    # Tank withdrawal cannot physically shrink: during shut-ins Np/Gp/Wp are
    # flat, but a smoothed-pressure uptick can still nudge Bo/Bg down a
    # fraction, producing a microscopic (non-physical) dip in F. Force it
    # monotonic non-decreasing before it reaches the H-O regression.
    F = np.maximum.accumulate(F)

    Eo = oil_and_dissolved_gas_expansion(Bo, Boi, Rsi, Rs, Bg)
    Eg = gas_cap_expansion(Boi, Bg, Bgi)
    Efw = rock_and_water_expansion(Boi, cf, cw, Swc, Pi, P)
    Et = combined_expansion(Eo, Eg, Efw, m)

    return pd.DataFrame({"P": P, "F": F, "Eo": Eo, "Eg": Eg, "Efw": Efw, "Et": Et})


def aquifer_constant_from_strength(strength):
    """Maps an aquifer strength label to its steady-state constant C (STB/psia)."""
    return AQUIFER_CONSTANTS[strength]


def compute_we(C, Pi, P):
    """
    Steady-state linear aquifer model.

    We = C * (Pi - P)
    where C is the aquifer constant in STB/psia
    """
    return C * (Pi - P)


def fit_zero_intercept(x_array, y_array):
    """
    Fits y = N * x through the origin using exact least-squares.
    No intercept allowed — zero expansion means zero withdrawal.

    Returns: N (slope = OOIP estimate), R2 (goodness of fit)
    """
    x = np.array(x_array)
    y = np.array(y_array)

    # Zero-intercept least-squares slope
    # Derivation: minimise sum((y - N*x)^2) -> dL/dN = 0 -> N = sum(x*y)/sum(x^2)
    N = np.dot(x, y) / np.dot(x, x)

    # R^2 for zero-intercept model (relative to mean, NOT relative to y=0)
    SS_res = np.sum((y - N * x) ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - (SS_res / SS_tot)

    return N, R2


def compute_drive_indices(ho_df, N_est, m_est, C_est, Pi):
    """
    Fraction of total underground withdrawal (F) attributable to each drive
    mechanism at every step, using the estimated N, m, C from the H-O
    analysis (Tab 3). Because F = N*Et + We = N*(Eo + m*Eg + Efw) + We,
    DDI + SDI + WDI sums to ~1.0 at every row -- exactly 1.0 if N_est/m_est/
    C_est were fit self-consistently against the same F.

    DDI (depletion drive index)  = N_est * (Eo + Efw) / F
    SDI (segregation/gas cap)    = N_est * m_est * Eg / F
    WDI (water drive index)      = C_est * (Pi - P) / F

    Returns a DataFrame with columns [DDI, SDI, WDI]. The first row (F = 0
    at initial conditions) is NaN in all three columns -- 0/0 is undefined,
    not zero.
    """
    F = ho_df["F"].to_numpy(dtype=float)
    Eo = ho_df["Eo"].to_numpy(dtype=float)
    Eg = ho_df["Eg"].to_numpy(dtype=float)
    Efw = ho_df["Efw"].to_numpy(dtype=float)
    P = ho_df["P"].to_numpy(dtype=float)

    total_energy = np.where(np.abs(F) > 1e-9, F, np.nan)

    DDI = (N_est * (Eo + Efw)) / total_energy
    SDI = (N_est * m_est * Eg) / total_energy
    WDI = (C_est * (Pi - P)) / total_energy

    return pd.DataFrame({"DDI": DDI, "SDI": SDI, "WDI": WDI})


def fit_joint_drive_parameters(ho_df, Pi):
    """
    Jointly fits N, m, C via a single non-negative least-squares (NNLS)
    regression against F, rather than the sequential 3-stage approach
    (Section 3A's Eo-only fit, then 3B's Et fit, then 3C's residual fit).

    F = N*Et + We = N*(Eo + m*Eg + Efw) + C*(Pi - P)
      = N*(Eo + Efw) + (N*m)*Eg + C*(Pi - P)

    Substituting Q = N*m makes this linear in (N, Q, C).

    The sequential approach compounds error across three separate
    regressions: the residual C fit inherits whatever bias is already baked
    into N and m from the first two. An unconstrained joint fit
    (np.linalg.lstsq) fixes the compounding but not a second problem: Eg
    and (Pi-P) are collinear (both increase monotonically through the
    production history), so an unconstrained solver can assign a negative
    coefficient to whichever term is redundant given the other -- e.g. a
    negative C "paid for" by an inflated N, or a negative Q showing up as a
    negative SDI for a case with no real gas cap. Physically, N, m, and C
    can never be negative (negative OOIP, negative gas cap ratio, and
    negative aquifer influx are all meaningless), so this is solved with
    scipy.optimize.nnls instead of unconstrained least squares: it
    minimizes the same ||A @ x - y||^2 objective subject to x >= 0.

    Returns: N_est, m_est, C_est, r2 (of the joint fit)
    """
    F = ho_df["F"].to_numpy(dtype=float)
    A = np.column_stack([
        (ho_df["Eo"] + ho_df["Efw"]).to_numpy(dtype=float),
        ho_df["Eg"].to_numpy(dtype=float),
        (Pi - ho_df["P"]).to_numpy(dtype=float),
    ])
    y = F

    coefficients, residual = nnls(A, y)
    N_est = coefficients[0]
    Q_est = coefficients[1]
    C_est = coefficients[2]
    m_est = Q_est / N_est if N_est > 0 else 0.0

    fitted = A @ coefficients
    SS_res = np.sum((y - fitted) ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - SS_res / SS_tot

    return N_est, m_est, C_est, r2
