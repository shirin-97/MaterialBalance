"""PVT correlations for the Havlena-Odeh diagnostic tool (spec v3, Section 4.1).

Standing's correlations apply only in the saturated region (P <= Pb).
Above bubble point, the reservoir is undersaturated: Rs is locked at its
bubble point value and Bo is modeled via isothermal compressibility.

scipy is intentionally not used anywhere in this module. Bubble point
pressure is computed via an explicit algebraic inverse of Standing's Rs
correlation -- no numerical root-finder is needed or permitted.
"""

import numpy as np
import pandas as pd


def compute_pb_from_rsi(Rsi, T, API, gamma_g):
    """
    Computes bubble point pressure algebraically from initial solution GOR.
    This is the direct inverse of Standing's Rs correlation.
    No root-finding or iteration required.

    Derivation:
        Standing's Rs: Rs = gamma_g * ((P/18.2 + 1.4) * 10^x)^1.205
        Solving for P: P = 18.2 * ((Rs/gamma_g)^(1/1.205) / 10^x - 1.4)

    Parameters:
        Rsi     : initial solution GOR (Scf/STB)
        T       : reservoir temperature (degF)
        API     : oil API gravity
        gamma_g : gas specific gravity

    Returns:
        Pb (psia)
    """
    x = 0.0125 * API - 0.00091 * T
    Pb = 18.2 * ((Rsi / gamma_g) ** (1 / 1.205) / (10 ** x) - 1.4)
    return Pb


def standing_Rs(P, T, API, gamma_g):
    """
    Standing's solution GOR correlation (saturated region only).

    Rs = gamma_g * ((P / 18.2 + 1.4) * 10^x)^1.205
    where x = 0.0125 * API - 0.00091 * T
    Units: Rs in Scf/STB, P in psia, T in degF
    Valid range: 100 < P < 5000 psia, 100F < T < 300F, 20 < API < 55
    """
    x = 0.0125 * API - 0.00091 * T
    Rs = gamma_g * ((P / 18.2 + 1.4) * 10 ** x) ** 1.205
    return Rs


def standing_Bo(Rs, T, API, gamma_g):
    """
    Standing's oil FVF correlation (saturated region only).

    Bo = 0.9759 + 1.2e-4 * (Rs * sqrt(gamma_g / gamma_o) + 1.25 * T)^1.2
    where gamma_o = 141.5 / (131.5 + API)
    Units: Bo in rb/STB
    """
    gamma_o = 141.5 / (131.5 + API)
    Bo = 0.9759 + 1.2e-4 * (Rs * np.sqrt(gamma_g / gamma_o) + 1.25 * T) ** 1.2
    return Bo


def bg_from_pressure(P, T, z=0.80):
    """
    Gas FVF.

    Bg = 0.00504 * z * (T + 460) / P
    Units: Bg in rb/Scf, T in degF, P in psia
    z is a simplified constant (default 0.80).
    """
    Bg = 0.00504 * z * (T + 460) / P
    return Bg


def get_pvt_at_pressure(P, Pb, Rsb, Bob, T, API, gamma_g, co=1.5e-5):
    """
    Returns Rs, Bo, Bg, Bw at a given pressure P.
    Enforces the saturated/undersaturated phase boundary.

    Parameters:
        P     : current pressure (psia)
        Pb    : bubble point pressure (psia)
        Rsb   : solution GOR at bubble point (Scf/STB)
        Bob   : oil FVF at bubble point (rb/STB)
        T     : reservoir temperature (degF)
        API   : oil API gravity
        gamma_g: gas specific gravity
        co    : isothermal compressibility of oil (1/psia), default 1.5e-5
    """

    if P >= Pb:
        # UNDERSATURATED REGION
        # Gas is locked in solution -- Rs stays constant at bubble point value
        Rs = Rsb

        # Oil compresses as pressure rises above Pb
        # Bo decreases slightly above Pb (oil is being compressed)
        Bo = Bob * np.exp(co * (Pb - P))

    else:
        # SATURATED REGION
        # Gas comes out of solution -- use Standing's correlations
        Rs = standing_Rs(P, T, API, gamma_g)
        Bo = standing_Bo(Rs, T, API, gamma_g)

    # Note: Free gas physically does not exist above Pb. Bg is computed here
    # as a hypothetical volume solely to prevent ZeroDivisionError or
    # TypeError in downstream material balance arrays. Do not change the Bg
    # math itself.
    Bg = bg_from_pressure(P, T)
    Bw = 1.03

    return Rs, Bo, Bg, Bw


def build_pvt_table(Pi, Pb, Rsb, Bob, T, API, gamma_g, co, pressure_array):
    """
    Builds the full PVT table across pressure_array. The first row
    corresponds to initial pressure Pi.

    Enforces the CRITICAL CONSTRAINT that Rs must never increase as
    pressure decreases below Pb: Rs[i] = min(Rs[i], Rs[i-1]).

    Returns a DataFrame with columns [P, Rs, Bo, Bg, Bw].
    """
    # Mathematically sanitize array: inject Pi, annihilate duplicates via set, sort descending
    pressures = list(set(list(pressure_array) + [Pi]))
    pressures.sort(reverse=True)

    rows = []
    prev_rs = None
    for P in pressures:
        Rs, Bo, Bg, Bw = get_pvt_at_pressure(P, Pb, Rsb, Bob, T, API, gamma_g, co)

        if prev_rs is not None:
            Rs = min(Rs, prev_rs)

        prev_rs = Rs
        rows.append({"P": P, "Rs": Rs, "Bo": Bo, "Bg": Bg, "Bw": Bw})

    return pd.DataFrame(rows, columns=["P", "Rs", "Bo", "Bg", "Bw"])


def interpolate_pvt_for_pressures(P_array, pvt_df):
    """
    Vectorized PVT property lookup for an arbitrary array of pressures
    (e.g. a production survey's P_avg column) against a reference PVT
    table, via linear interpolation.

    This is deliberately NOT Standing's correlation (get_pvt_at_pressure)
    and NOT a pandas merge/join: production survey pressures essentially
    never exactly match the PVT table's lab-sampled (or synthetically
    generated) pressures, so an exact-key join would drop almost every
    row. Interpolating against the reference table is also the physically
    correct choice for uploaded lab PVT data specifically -- real
    measured fluid properties are preferable to a generic correlation
    estimate (see Section 14 of the spec).

    Returns a DataFrame with columns [Rs, Bo, Bg, Bw], aligned index-for-
    index with P_array.
    """
    pvt_sorted = pvt_df.sort_values("P")
    P_array = np.asarray(P_array, dtype=float)
    return pd.DataFrame({
        "Rs": np.interp(P_array, pvt_sorted["P"], pvt_sorted["Rs"]),
        "Bo": np.interp(P_array, pvt_sorted["P"], pvt_sorted["Bo"]),
        "Bg": np.interp(P_array, pvt_sorted["P"], pvt_sorted["Bg"]),
        "Bw": np.interp(P_array, pvt_sorted["P"], pvt_sorted["Bw"]),
    })
