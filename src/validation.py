"""Data validation and flagging (spec v6, Section 4.6).

validate_inputs() runs 8 checks and returns a list of structured
ValidationMessage records (severity + message + optional row index) so the
UI can colorize banners and decide whether to gate analysis, rather than
parsing the message text.

The original Section 4.6 spec listed a 9th check ("F_remaining after Wp
subtraction is negative") validating the forward synthetic generator's
internal energy budget (F_synthetic = N*(Eo+m*Eg+Efw)+We). That budget
depends on N, m, and We, none of which validate_inputs(production_df,
pvt_df) has access to -- and reconstructing F_remaining from the *observed*
F = Np*Bo + (Gp-Np*Rs)*Bg + Wp*Bw instead just cancels Wp back out
algebraically (F - Wp*Bw = Np*Bo + (Gp-Np*Rs)*Bg, independent of Wp's
magnitude), so it can never actually detect "water production exceeds
total energy." Removed rather than kept as a check that cannot fire on the
condition it claims to check.
"""

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from .pvt import interpolate_pvt_for_pressures


class ValidationMessage(NamedTuple):
    severity: str  # "ERROR" or "WARNING"
    message: str
    row: Optional[int] = None


def has_blocking_errors(messages):
    """True if any message is an ERROR. Errors block analysis; warnings do not."""
    return any(m.severity == "ERROR" for m in messages)


def _pvt_lookup(production_df, pvt_df):
    """Interpolates Rs, Bo, Bg, Bw from pvt_df onto each production row's P_avg."""
    pvt_at_P = interpolate_pvt_for_pressures(production_df["P_avg"], pvt_df)
    return (pvt_at_P["Rs"].to_numpy(), pvt_at_P["Bo"].to_numpy(),
            pvt_at_P["Bg"].to_numpy(), pvt_at_P["Bw"].to_numpy())


# Metric -> Imperial conversion scalars, applied to raw production data only
# (the mapped PVT table is assumed already in Imperial units -- see
# sanitize_mapped_data's docstring for why this is a scoped assumption, not
# an oversight).
METRIC_TO_IMPERIAL = {
    "Np": 6.2898,    # Sm3 oil -> STB
    "Wp": 6.2898,    # Sm3 water -> STB
    "Gp": 35.315,    # Sm3 gas -> Scf
    "P_avg": 14.5038,  # Bar -> psia
}


def sanitize_mapped_data(production_df_raw, pvt_df_raw, production_mapping, pvt_mapping,
                          is_incremental=False, is_metric=False):
    """
    Turns a raw uploaded file plus a user-confirmed column mapping into the
    engine's standard DataFrames.

    production_mapping / pvt_mapping: dicts of {standard_name: user_column_name},
    e.g. {"Np": "Cum Oil (stb)", "P_avg": "Reservoir Pressure", ...}.

    is_incremental: True if the mapped Np/Gp/Wp columns are daily/monthly
    incremental rates rather than running cumulative totals -- converted to
    cumulative via .cumsum() (see app.py's smart cumulative detection,
    which defaults this from Np's monotonicity before the user confirms).

    is_metric: True if the mapped production data is in metric units
    (Sm3, Bar) rather than the engine's required Imperial units (STB, Scf,
    psia) -- converted via METRIC_TO_IMPERIAL. Only the production columns
    are converted, not the PVT table: the PVT table is assumed to already
    be in Imperial units (e.g. a lab report or this app's own template),
    independent of what units the production data happens to come in.
    If your PVT table is *also* in metric units, it needs converting
    separately before upload -- this function does not detect that.

    Steps (deliberately ruthless -- real uploaded field data is never clean):
    1. Rename the user's columns to the engine's standard names and drop
       anything not in the mapping.
    2. Coerce every numeric column to numeric, invalid parses -> NaN
       (Date is parsed as a date instead, invalid parses -> NaT).
    3. If is_metric, scale Np/Wp (Sm3->STB), Gp (Sm3->Scf), and P_avg
       (Bar->psia) up to Imperial -- before any of the steps below, since
       cumsum/the NaN-or-zero drop/interpolation all assume Imperial units
       already.
    4. Sort the production data strictly by Date, ascending -- cumsum (next
       step) is meaningless out of chronological order, and a source file
       is not guaranteed to already be sorted.
    5. If is_incremental, cumsum Np/Gp/Wp to convert incremental rates to
       running cumulative totals.
    6. If is_incremental, smooth P_avg with a 30-day rolling average (zeros
       treated as missing first, so shut-in days don't drag it down) --
       daily gauge pressure fluctuates with choke size/flow rate, not just
       the underlying depletion trend, which otherwise breaks the tank
       model's monotonic-withdrawal (F) assumption.
    7. Drop production rows where Np or P_avg is NaN or exactly 0 -- these
       can't contribute to the material balance and typically mark shut-in
       or missing-data days. (After step 6, this rarely drops any P_avg
       rows for incremental data -- smoothing fills them instead.)
    8. Sort the PVT table descending by P, regardless of how the source
       file was ordered -- build_pvt_table/interpolate_pvt_for_pressures
       assume this, and a user's spreadsheet export is not guaranteed to.

    Returns: (production_df, pvt_df), ready for validate_inputs() /
    compute_ho_terms().
    """
    production_df = production_df_raw.rename(
        columns={user_col: standard for standard, user_col in production_mapping.items()}
    )[list(production_mapping.keys())].copy()

    pvt_df = pvt_df_raw.rename(
        columns={user_col: standard for standard, user_col in pvt_mapping.items()}
    )[list(pvt_mapping.keys())].copy()

    if "Date" in production_df.columns:
        production_df["Date"] = pd.to_datetime(production_df["Date"], errors="coerce")

    for col in production_df.columns:
        if col != "Date":
            production_df[col] = pd.to_numeric(production_df[col], errors="coerce")

    for col in pvt_df.columns:
        pvt_df[col] = pd.to_numeric(pvt_df[col], errors="coerce")

    if is_metric:
        for col, scalar in METRIC_TO_IMPERIAL.items():
            if col in production_df.columns:
                production_df[col] = production_df[col] * scalar

    if "Date" in production_df.columns:
        production_df = production_df.sort_values("Date", ascending=True).reset_index(drop=True)

    if is_incremental:
        for col in ("Np", "Gp", "Wp"):
            if col in production_df.columns:
                production_df[col] = production_df[col].cumsum()

        # Daily pressure gauges fluctuate with choke size / flow rate, not just
        # the underlying reservoir depletion trend -- that noise breaks the
        # tank model's monotonic-withdrawal (F) assumption. Smooth it with a
        # 30-day rolling average before it reaches get_pvt_at_pressure or the
        # dropna/zero purge. Zeros first, since shut-in days would otherwise
        # drag the rolling average down toward zero rather than being ignored.
        if "P_avg" in production_df.columns:
            production_df["P_avg"] = production_df["P_avg"].replace(0, np.nan)
            production_df["P_avg"] = production_df["P_avg"].rolling(window=30, min_periods=1).mean()
            production_df["P_avg"] = production_df["P_avg"].ffill().bfill()

    production_df = production_df[
        production_df["Np"].notna() & (production_df["Np"] != 0)
        & production_df["P_avg"].notna() & (production_df["P_avg"] != 0)
    ].reset_index(drop=True)

    pvt_df = pvt_df.sort_values("P", ascending=False).reset_index(drop=True)

    return production_df, pvt_df


def validate_inputs(production_df, pvt_df):
    """
    Runs all 8 validation checks against production_df and pvt_df.

    Returns a list of ValidationMessage records. Errors block analysis;
    warnings allow proceeding with a visible note.
    """
    messages = []

    # Pb isn't passed to validate_inputs, so it's inferred from pvt_df itself:
    # Rs is locked at Rsb for the entire undersaturated region (P >= Pb) and
    # strictly lower everywhere in the saturated region (P < Pb), so
    # Rsb = max(Rs) identifies the bubble point regardless of whether the
    # table includes any undersaturated rows at all.
    pvt_sorted_desc = pvt_df.sort_values("P", ascending=False).reset_index(drop=True)
    Rs_series = pvt_sorted_desc["Rs"].to_numpy()
    Rsb = Rs_series.max() if len(Rs_series) > 0 else 0.0

    # 1. Rs increases as P decreases below Pb (within pvt_df, saturated region)
    Rs_below_pb = Rs_series[Rs_series < Rsb - 1e-9]
    if len(Rs_below_pb) > 1 and np.any(np.diff(Rs_below_pb) > 1e-9):
        messages.append(ValidationMessage(
            "ERROR", "PVT table physically invalid — Rs must not increase below bubble point"
        ))

    # 2. Bo above Pb does not follow compressibility trend (Bo should decrease as P rises above Pb)
    undersaturated = pvt_sorted_desc[pvt_sorted_desc["Rs"] >= Rsb - 1e-9].sort_values("P")
    if len(undersaturated) > 1:
        Bo_undersaturated = undersaturated["Bo"].to_numpy()
        if np.any(np.diff(Bo_undersaturated) > 1e-9):
            messages.append(ValidationMessage("WARNING", "Check undersaturated Bo values"))

    # 3. Np non-monotonic
    Np = production_df["Np"].to_numpy(dtype=float)
    for i in range(1, len(Np)):
        if Np[i] < Np[i - 1] - 1e-9:
            messages.append(ValidationMessage("WARNING", f"Non-monotonic Np at row {i}", row=i))

    # 4. Gp non-monotonic
    Gp = production_df["Gp"].to_numpy(dtype=float)
    for i in range(1, len(Gp)):
        if Gp[i] < Gp[i - 1] - 1e-9:
            messages.append(ValidationMessage("WARNING", f"Non-monotonic Gp at row {i}", row=i))

    Rs, Bo, Bg, Bw = _pvt_lookup(production_df, pvt_df)
    Wp = production_df["Wp"].to_numpy(dtype=float)

    # 5. (Gp - Np*Rs) negative at any row
    free_gas = Gp - Np * Rs
    for i in np.where(free_gas < -1e-6)[0]:
        messages.append(ValidationMessage(
            "WARNING", f"Producing GOR below solution GOR at row {i}", row=int(i)
        ))

    # 6. F non-monotonic
    F = Np * Bo + free_gas * Bg + Wp * Bw
    for i in range(1, len(F)):
        if F[i] < F[i - 1] - 1e-6:
            messages.append(ValidationMessage(
                "WARNING", f"Non-physical withdrawal decrease at row {i}", row=i
            ))

    # 7. Fewer than 8 data points
    if len(production_df) < 8:
        messages.append(ValidationMessage(
            "WARNING", "H-O analysis unreliable with fewer than 8 pressure points"
        ))

    # 8. Pressure data missing for more than 30% of rows
    missing_fraction = production_df["P_avg"].isna().mean() if len(production_df) > 0 else 1.0
    if missing_fraction > 0.30:
        messages.append(ValidationMessage(
            "ERROR", "Insufficient pressure data for material balance"
        ))

    return messages
