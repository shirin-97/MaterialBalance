"""Havlena-Odeh Reservoir Drive Mechanism Diagnostic Tool (spec v11, Section 5).

Architecture notes (see also the changelog):
- The sidebar is rendered BEFORE the tabs so it always renders even when a
  blocking validation error later calls st.stop() inside a tab body.
- Buttons that need to write to session_state keys backing already-rendered
  sidebar widgets ("Auto-find best m", "Confirm Mapping & Run Analysis") use
  on_click callbacks rather than the literal `if st.button(...):
  st.session_state[...] = ...` pattern -- Streamlit raises
  StreamlitAPIException if you write to a widget's key after that widget
  has been instantiated in the same run; callbacks run before the widget is
  (re)instantiated, so this is the robust way to get the same
  survives-reruns behavior.
- ho_df is computed once, before st.tabs() is called, from already-loaded
  production_df/pvt_df. Section 3B's "Auto-find best m" and the m slider
  itself only do cheap vector algebra on that pre-computed ho_df.
- The Synthetic Data Generator UI has been removed (see the changelog) --
  the app now only accepts real uploaded data via a dynamic column mapper.
  src/synthetic.py and its test suite are untouched; generate_field_case is
  simply no longer called from this file.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.material_balance import (
    compute_drive_indices,
    compute_ho_terms,
    fit_joint_drive_parameters,
    fit_zero_intercept,
)
from src.plots import (
    plot_drive_indices,
    plot_f_vs_eo,
    plot_f_vs_et,
    plot_gor_vs_np,
    plot_pressure_vs_np,
    plot_production_history,
    plot_water_drive_diagnostic,
    plot_wor_vs_np,
)
from src.pvt import interpolate_pvt_for_pressures
from src.validation import has_blocking_errors, sanitize_mapped_data, validate_inputs

st.set_page_config(page_title="Havlena-Odeh Diagnostic Tool", layout="wide")

TEMPLATE_DIR = "data/sample_templates"

PRODUCTION_FIELDS = ["Date", "Np", "Gp", "Wp", "P_avg"]
PVT_FIELDS = ["P", "Rs", "Bo", "Bg", "Bw"]

# Common real-world naming variants (e.g. Volve-style exports) to smart-default
# the mapping selectboxes to, so the user usually only has to confirm, not hunt.
FIELD_ALIASES = {
    "Date": ["date", "prod_date", "production date"],
    "Np": ["np", "cum oil", "cumulative oil", "oil cum", "oil_cum_stb"],
    "Gp": ["gp", "cum gas", "cumulative gas", "gas cum", "gas_cum_scf"],
    "Wp": ["wp", "cum water", "cumulative water", "water cum", "water_cum_stb"],
    "P_avg": ["p_avg", "pressure", "avg pressure", "reservoir pressure", "bhp"],
    "P": ["p", "pressure", "pressure (psia)"],
    "Rs": ["rs", "gor", "solution gor"],
    "Bo": ["bo", "oil fvf"],
    "Bg": ["bg", "gas fvf"],
    "Bw": ["bw", "water fvf"],
}

WELL_COLUMN_NAME_HINTS = ["well", "wellbore", "uwi", "api14", "npd"]


def _detect_well_column(df):
    """
    Heuristically finds a categorical well-identifier column (e.g.
    WELL_BORE_CODE, as in Volve) worth filtering to a single well:
    string-like dtype (covers both legacy object-dtype columns and
    pandas' newer dedicated string dtype), and moderate cardinality (more
    than one distinct value, but not so many that it's free text /
    effectively unique per row, like a Date string). Returns None if no
    such column exists (single-well file).
    """
    candidates = [
        col for col in df.columns
        if pd.api.types.is_string_dtype(df[col])
        and 1 < df[col].nunique(dropna=True) < len(df)
    ]
    if not candidates:
        return None
    for col in candidates:
        if any(hint in str(col).lower() for hint in WELL_COLUMN_NAME_HINTS):
            return col
    return candidates[0]

# ---------------------------------------------------------------------------
# Session state defaults (set before any widget with these keys is created)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "Pi_input": 3000.0,
    "Pb_input": 2500.0,
    "Boi_input": 1.20,
    "Bob_input": 1.25,
    "Rsi_input": 600.0,
    "Rsb_input": 600.0,
    "Bgi_input": 0.00090,
    "co_input": 1.5e-5,
    "Swc_input": 0.20,
    "cf_input": 4e-6,
    "cw_input": 3e-6,
    "Bw_input": 1.03,
    "m_slider": 0.0,
    "best_m": 0.0,
    "best_m_r2": None,
    "production_df": None,
    "pvt_df": None,
    "true_params": None,
    "data_mode": None,
    "_ho_df_cache": None,
    "raw_production_df": None,
    "raw_pvt_df": None,
    "raw_production_file_id": None,
    "raw_pvt_file_id": None,
    "well_filtered_production_df": None,
    "incremental_checkbox": False,
    "_last_np_mapping_for_checkbox": None,
    "data_units": "Metric (Sm3, Bar)",
}
for _key, _value in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


# ---------------------------------------------------------------------------
# Raw file loading (Action 2.2: store the raw DataFrame in session_state so
# the file isn't re-read on every widget click -- only re-parsed when the
# uploaded file's identity actually changes).
# ---------------------------------------------------------------------------
def _sync_raw_file(uploaded_file, df_key, id_key):
    if uploaded_file is None:
        return
    if st.session_state.get(id_key) == uploaded_file.file_id:
        return
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read {uploaded_file.name}: {e}")
        return
    st.session_state[df_key] = df
    st.session_state[id_key] = uploaded_file.file_id


def _normalize_column_name(name):
    """Lowercase, and treat underscores/hyphens as spaces, so "Gas_FVF" and
    "gas fvf" compare equal -- real headers mix these conventions freely
    (e.g. dummy_pvt.xlsx uses "Gas_FVF", "Water_FVF", "Solution_GOR" while
    FIELD_ALIASES was originally written with spaces)."""
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _guess_column(field, columns):
    """Smart-default a mapping selectbox: exact (case-insensitive,
    underscore/space-insensitive) name match first, then a substring match
    against known aliases (real headers are rarely an exact match -- e.g.
    "Cum Oil (stb)" for Np), else fall back to the first column so the
    selectbox always has a valid initial value."""
    normalized = {_normalize_column_name(c): c for c in columns}
    candidates = [_normalize_column_name(field)] + [
        _normalize_column_name(a) for a in FIELD_ALIASES.get(field, [])
    ]

    for cand in candidates:
        if cand in normalized:
            return normalized[cand]

    for cand in candidates:
        if len(cand) < 3:
            continue  # too short to substring-match without false positives
        for norm_name, original in normalized.items():
            if cand in norm_name:
                return original

    return columns[0] if len(columns) > 0 else None


def _attach_rate_columns(production_df):
    """
    plot_production_history (Tab 2 Plot 1) reads oil_rates_noisy/
    gas_rates_noisy/water_rates_noisy directly rather than differentiating
    cumulatives, per spec Section 5 -- that requirement exists to avoid
    artificial flatline/spike artifacts from differentiating a noise-
    injected *synthetic* cumulative (see src/synthetic.py's
    inject_noise_via_rates). Real uploaded data has no such pipeline and no
    such column, so it's derived here via plain differentiation instead.
    Floored at zero: a validation warning (not an error) already flags
    non-monotonic Np/Gp/Wp, but the plot still needs to render.
    """
    production_df = production_df.copy()
    for cum_col, rate_col in [("Np", "oil_rates_noisy"), ("Gp", "gas_rates_noisy"), ("Wp", "water_rates_noisy")]:
        production_df[rate_col] = np.maximum(np.diff(production_df[cum_col].to_numpy(), prepend=0), 0)
    return production_df


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _on_confirm_mapping():
    production_mapping = {
        field: st.session_state[f"map_prod_{field}"] for field in PRODUCTION_FIELDS
    }
    pvt_mapping = {
        field: st.session_state[f"map_pvt_{field}"] for field in PVT_FIELDS
    }

    production_df, pvt_df = sanitize_mapped_data(
        st.session_state["well_filtered_production_df"], st.session_state["raw_pvt_df"],
        production_mapping, pvt_mapping,
        is_incremental=bool(st.session_state["incremental_checkbox"]),
        is_metric=(st.session_state["data_units"] == "Metric (Sm3, Bar)"),
    )
    production_df = _attach_rate_columns(production_df)

    st.session_state["production_df"] = production_df
    st.session_state["pvt_df"] = pvt_df
    st.session_state["true_params"] = None
    st.session_state["data_mode"] = "upload"


def _on_autofind_m():
    ho_df = st.session_state.get("_ho_df_cache")
    if ho_df is None:
        return
    m_values = np.arange(0.0, 1.01, 0.01)
    r2_values = []
    for m_try in m_values:
        Et_try = ho_df["Eo"] + m_try * ho_df["Eg"] + ho_df["Efw"]
        _, r2 = fit_zero_intercept(Et_try.values, ho_df["F"].values)
        r2_values.append(r2)
    best_idx = int(np.argmax(r2_values))
    st.session_state["best_m"] = float(m_values[best_idx])
    st.session_state["best_m_r2"] = float(r2_values[best_idx])
    st.session_state["m_slider"] = st.session_state["best_m"]


# ---------------------------------------------------------------------------
# Sidebar (rendered first so it always shows, even if a tab later calls
# st.stop() on a blocking validation error)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Reservoir Constants")
    st.number_input("Pb (psia) — bubble point", key="Pb_input")

    with st.expander("⚙️ Advanced Reservoir Constants", expanded=False):
        st.number_input("Pi (psia)", key="Pi_input")
        st.number_input("Boi (rb/STB)", key="Boi_input", format="%.4f")
        st.number_input("Bob (rb/STB) — Bo at bubble point", key="Bob_input", format="%.4f")
        st.number_input("Rsi (Scf/STB)", key="Rsi_input")
        st.number_input("Rsb (Scf/STB) — Rs at bubble point", key="Rsb_input")
        st.number_input("Bgi (rb/Scf)", key="Bgi_input", format="%.6f")
        st.number_input("co (1/psia) — undersaturated oil compressibility", key="co_input", format="%.2e")
        st.number_input("Swc (fraction)", key="Swc_input", min_value=0.0, max_value=1.0)
        st.number_input("cf (1/psia)", key="cf_input", format="%.2e")
        st.number_input("cw (1/psia)", key="cw_input", format="%.2e")
        st.number_input("Bw (rb/STB)", key="Bw_input", format="%.4f")

    st.header("Manual Visualization (Tab 3B)")
    st.slider("m (gas cap ratio)", 0.0, 1.0, step=0.05, key="m_slider")
    st.caption(
        "Note: This slider is only for visually testing the straight-line fit "
        "on Tab 3B. The final Summary Card and Drive Indices use the "
        "automated NNLS solver."
    )
    if st.session_state["best_m_r2"] is not None:
        st.info(
            f"Best fit m = {st.session_state['best_m']:.2f} → "
            f"R² = {st.session_state['best_m_r2']:.3f}"
        )

    st.header("Assumptions & Limitations")
    with st.expander("Show assumptions"):
        st.markdown(
            """
1. **PVT** — Standing's correlations apply only in the saturated region
   (P ≤ Pb). Valid range: 100–5000 psia, 100–300°F, 20–55 API.
   Outside these ranges, accuracy degrades.
2. Undersaturated region uses constant isothermal compressibility (co).
   For highly compressible oils or large pressure ranges above Pb,
   a pressure-dependent co is more accurate.
3. Aquifer model is steady-state linear (We = C·(Pi−P)).
   Underestimates influx for large aquifers or early-time transient
   behavior. Van Everdingen-Hurst is the rigorous alternative.
4. Gas FVF uses z = 0.80 constant. For high-pressure, high-temperature,
   or sour gas, use Hall-Yarborough or Pitzer correlations.
5. Material balance assumes tank model (uniform pressure throughout
   reservoir). Does not account for pressure gradients.
6. H-O analysis requires representative average reservoir pressure.
   Pseudo-steady-state must be achieved before each pressure measurement
   for the tank model assumption to hold.
7. Zero-intercept regression is enforced by design. Any apparent
   intercept in real data is attributed to data quality issues,
   not a physical phenomenon.
            """
        )

    st.header("About")
    st.markdown(
        "Implements the Havlena-Odeh straight-line method to diagnose the "
        "primary drive mechanism of a reservoir (depletion, gas cap, water "
        "drive) from historical production and pressure data.\n\n"
        "[Equinor Volve data](https://www.equinor.com/energy/volve-data-sharing)"
    )

# ---------------------------------------------------------------------------
# Data availability + validation (computed once, shared by every tab)
# ---------------------------------------------------------------------------
production_df = st.session_state["production_df"]
pvt_df = st.session_state["pvt_df"]
data_available = production_df is not None and pvt_df is not None

messages = validate_inputs(production_df, pvt_df) if data_available else []
blocking = has_blocking_errors(messages) if data_available else False
errors_only = [m for m in messages if m.severity == "ERROR"]

# All data is real uploaded data now (Synthetic Data Generator UI removed),
# so validation warnings (e.g. "Producing GOR below solution GOR") are always
# a genuine, actionable data-quality signal worth showing.
warnings_only = [m for m in messages if m.severity == "WARNING"]


def _render_validation_banners():
    for m in errors_only:
        st.error(m.message)
    for m in warnings_only:
        st.warning(m.message)


# ---------------------------------------------------------------------------
# Action 3: compute ho_df once, before st.tabs(), using the current sidebar
# constants (as set by the user in the sidebar, or left at their defaults).
# ---------------------------------------------------------------------------
ho_df = None
indices_df = None
drive_params = None  # (N_est, m_est, C_est, r2) from the joint fit, for Tab 4
if data_available and not blocking:
    Pi_current = st.session_state["Pi_input"]
    constants = dict(
        Pi=Pi_current,
        Boi=st.session_state["Boi_input"],
        Bgi=st.session_state["Bgi_input"],
        Rsi=st.session_state["Rsi_input"],
        cf=st.session_state["cf_input"],
        cw=st.session_state["cw_input"],
        Swc=st.session_state["Swc_input"],
        m=st.session_state["m_slider"],
    )
    ho_df = compute_ho_terms(production_df, pvt_df, constants)

    # Tab 4's drive indices need N, m, C fit jointly (not Tab 3's sequential
    # 3A -> 3B -> 3C fits) so DDI + SDI + WDI actually sums close to 1.0 --
    # see fit_joint_drive_parameters' docstring for why the sequential
    # approach compounds error for aquifer-influenced cases.
    N_est, m_est, C_est, r2_joint = fit_joint_drive_parameters(ho_df, Pi_current)
    drive_params = (N_est, m_est, C_est, r2_joint)
    indices_df = compute_drive_indices(ho_df, N_est, m_est, C_est, Pi_current)
st.session_state["_ho_df_cache"] = ho_df


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Havlena-Odeh Reservoir Drive Mechanism Diagnostic Tool")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Input", "Production Diagnostics", "Havlena-Odeh Analysis", "Insights & Drive Indices"]
)

# --- TAB 1: Data Input -------------------------------------------------------
# Synthetic Data Generator UI removed -- Upload Mode is now the only entry
# point (src/synthetic.py and its tests are untouched; generate_field_case is
# simply not called from this file anymore).
#
# Strict step-by-step gating: Step 2 (well filtering) never renders until
# Step 1 (both files uploaded) is done; Step 3 (column mapping) never
# renders until Step 2 is resolved (either a well is selected, or no well
# identifier column exists to filter on). Earlier steps stay visible
# (re-uploading a file is a normal thing to want to do), but nothing past
# the current unmet step ever renders.
with tab1:
    st.subheader("Step 1: Upload Files")

    prod_file = st.file_uploader(
        "Upload production file", type=["csv", "xlsx", "xls"], key="prod_uploader"
    )
    pvt_file = st.file_uploader(
        "Upload PVT file", type=["csv", "xlsx", "xls"], key="pvt_uploader"
    )
    st.caption(
        "Production file required physical parameters: Date, Np (Cumulative Oil), "
        "Gp (Cumulative Gas), Wp (Cumulative Water), P_avg (Average Pressure).\n\n"
        "PVT file required physical parameters: P (Pressure), Rs (Solution GOR), "
        "Bo (Oil FVF), Bg (Gas FVF), Bw (Water FVF).\n\n"
        "Column names don't need to match exactly -- map them in Step 3."
    )

    try:
        with open(f"{TEMPLATE_DIR}/production_template.csv", "rb") as f:
            prod_template_bytes = f.read()
        with open(f"{TEMPLATE_DIR}/pvt_template.csv", "rb") as f:
            pvt_template_bytes = f.read()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download production template", prod_template_bytes,
                file_name="production_template.csv", mime="text/csv",
            )
        with col2:
            st.download_button(
                "Download PVT template", pvt_template_bytes,
                file_name="pvt_template.csv", mime="text/csv",
            )
    except FileNotFoundError:
        pass

    _sync_raw_file(prod_file, "raw_production_df", "raw_production_file_id")
    _sync_raw_file(pvt_file, "raw_pvt_df", "raw_pvt_file_id")

    raw_production_df = st.session_state["raw_production_df"]
    raw_pvt_df = st.session_state["raw_pvt_df"]

    if raw_production_df is None or raw_pvt_df is None:
        st.info("Upload both a production file and a PVT file to continue.")
    else:
        # --- Step 2: Well Filtering ---
        well_col = _detect_well_column(raw_production_df)
        if well_col is not None:
            st.subheader("Step 2: Select Well")
            wells = sorted(raw_production_df[well_col].dropna().unique().tolist())
            st.selectbox(
                f"Filter to a single well (detected identifier column: {well_col})",
                wells, key="selected_well",
            )
            well_filtered_df = raw_production_df[
                raw_production_df[well_col] == st.session_state["selected_well"]
            ].reset_index(drop=True)
        else:
            well_filtered_df = raw_production_df
        st.session_state["well_filtered_production_df"] = well_filtered_df

        # --- Step 3: Column Mapping ---
        st.subheader("Step 3: Map Columns")
        st.caption(
            "Match each required physical parameter to the column that holds "
            "it in your uploaded file."
        )
        prod_cols = list(well_filtered_df.columns)
        pvt_cols = list(raw_pvt_df.columns)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Production Requirements**")
            for field in PRODUCTION_FIELDS:
                key = f"map_prod_{field}"
                if key not in st.session_state:
                    st.session_state[key] = _guess_column(field, prod_cols)
                st.selectbox(field, prod_cols, key=key)
        with col2:
            st.markdown("**PVT Requirements**")
            for field in PVT_FIELDS:
                key = f"map_pvt_{field}"
                if key not in st.session_state:
                    st.session_state[key] = _guess_column(field, pvt_cols)
                st.selectbox(field, pvt_cols, key=key)

        # Smart cumulative detection: re-defaults the checkbox whenever the
        # Np mapping changes, but leaves the user's manual choice alone
        # otherwise (see _DEFAULTS / _last_np_mapping_for_checkbox).
        np_col = st.session_state["map_prod_Np"]
        is_cumulative = pd.to_numeric(
            well_filtered_df[np_col], errors="coerce"
        ).dropna().is_monotonic_increasing
        if st.session_state["_last_np_mapping_for_checkbox"] != np_col:
            st.session_state["incremental_checkbox"] = not is_cumulative
            st.session_state["_last_np_mapping_for_checkbox"] = np_col
        st.checkbox(
            "My production data is daily/monthly incremental rates "
            "(Auto-calculate cumulative totals)",
            key="incremental_checkbox",
        )

        st.radio(
            "Data Units",
            ["Imperial (STB, Scf, psi)", "Metric (Sm3, Bar)"],
            key="data_units",
        )
        st.caption(
            "The Havlena-Odeh engine requires Imperial units (STB, Scf, psia). "
            "If Metric, Np/Wp (Sm3→STB), Gp (Sm3→Scf), and P_avg (Bar→psia) "
            "are converted automatically. The PVT table is assumed to already "
            "be in Imperial units regardless of this setting."
        )

        st.button("Confirm Mapping & Run Analysis", on_click=_on_confirm_mapping)

    if data_available:
        st.subheader("Production data preview (first 10 rows)")
        st.dataframe(production_df.head(10))
        st.subheader("PVT table preview")
        st.dataframe(pvt_df.head(10))

        st.subheader("PVT Interpolation Preview")
        st.caption(
            "Rs, Bo, Bg interpolated from the mapped PVT table onto each "
            "production row's P_avg (linear interpolation, not a merge/join "
            "on exact pressure -- production and PVT-sample pressures almost "
            "never match exactly)."
        )
        interpolated_preview = interpolate_pvt_for_pressures(production_df["P_avg"], pvt_df)
        st.dataframe(pd.concat(
            [production_df[["P_avg"]].reset_index(drop=True), interpolated_preview[["Rs", "Bo", "Bg"]]],
            axis=1,
        ).head(10))

        st.subheader("Validation")
        if not messages:
            st.success("All validation checks passed.")
        else:
            _render_validation_banners()

# --- TAB 2: Production Diagnostics ------------------------------------------
with tab2:
    if not data_available:
        st.info("Upload production and PVT data in Tab 1 first.")
    elif blocking:
        st.error("Blocking validation errors — see banners below. Analysis cannot proceed.")
        _render_validation_banners()
    else:
        if warnings_only:
            _render_validation_banners()
        st.subheader("Production Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_production_history(production_df), width='stretch')
            with st.expander("Interpretation guide"):
                st.markdown(
                    "Mark visible inflection points in oil rate, GOR, and WOR "
                    "to spot the onset of gas or water breakthrough."
                )
            st.plotly_chart(plot_wor_vs_np(production_df), width='stretch')
            with st.expander("Interpretation guide"):
                st.markdown(
                    "- Straight line on semi-log → Well-behaved waterflood or aquifer\n"
                    "- Upward curvature → Channeling or heterogeneous breakthrough\n"
                    "- Zero throughout early life → No water influx initially"
                )
        with col2:
            st.plotly_chart(plot_gor_vs_np(production_df), width='stretch')
            with st.expander("Interpretation guide"):
                st.markdown(
                    "- Flat then rising sharply → Solution gas drive\n"
                    "- Rising from early time → Gas cap or early gas breakthrough\n"
                    "- Relatively flat throughout → Strong water drive maintaining pressure above Pb"
                )
            st.plotly_chart(
                plot_pressure_vs_np(production_df, st.session_state["Pb_input"]),
                width='stretch',
            )
            with st.expander("Interpretation guide"):
                st.markdown(
                    "- Steep decline → Depletion, poor pressure support\n"
                    "- Gentle decline → Strong aquifer or active injection\n"
                    "- Pressure staying above Pb → Undersaturated behavior throughout"
                )

# --- TAB 3: Havlena-Odeh Analysis -------------------------------------------
with tab3:
    if not data_available:
        st.info("Upload production and PVT data in Tab 1 first.")
    elif blocking:
        st.error("Blocking validation errors — see banners below. Analysis cannot proceed.")
        _render_validation_banners()
    else:
        if warnings_only:
            _render_validation_banners()

        Pi = st.session_state["Pi_input"]
        m = st.session_state["m_slider"]

        # Depletion Diagnostic Plot (F vs Eo)
        st.subheader("Depletion Diagnostic Plot (F vs Eo)")
        N_fit_eo, r2_eo = fit_zero_intercept(ho_df["Eo"].values, ho_df["F"].values)
        st.plotly_chart(plot_f_vs_eo(ho_df["Eo"].values, ho_df["F"].values, N_fit_eo, r2_eo),
                         width='stretch')
        st.markdown(
            "**Equation:** F = N * Eo\n\n"
            "R² > 0.95, random scatter → Strong depletion drive. OOIP estimate reliable.\n\n"
            "Points curve upward (above line) → Additional energy source present. "
            "Check the Gas Cap Diagnostic Plot (if GOR rising) or Water Drive Diagnostic "
            "Plot (if WOR rising).\n\n"
            "Points curve downward (below line) → Check data quality. Possibly "
            "high rock/water compressibility not accounted for in Efw."
        )

        # Gas Cap Diagnostic Plot (F vs Et)
        st.subheader("Gas Cap Diagnostic Plot (F vs Et)")
        N_fit_et, r2_et = fit_zero_intercept(ho_df["Et"].values, ho_df["F"].values)
        st.plotly_chart(plot_f_vs_et(ho_df["Et"].values, ho_df["F"].values, N_fit_et, r2_et, m),
                         width='stretch')
        st.button("Auto-find best m", on_click=_on_autofind_m)
        st.markdown(
            "**Equation:** F = N * (Eo + m * Eg + Efw)\n\n"
            "Best-fit m > 0.05 AND R² improves vs depletion → Gas cap contribution detected.\n\n"
            "Best-fit m ≈ 0 AND R² unchanged → No evidence of gas cap. "
            "Check the Water Drive Diagnostic Plot if depletion R² < 0.90."
        )

        # Water Drive Diagnostic Plot ((F - N*Et) vs (Pi - P))
        st.subheader("Water Drive Diagnostic Plot ((F - N*Et) vs (Pi - P))")
        pi_minus_p = Pi - ho_df["P"].values
        f_residual = ho_df["F"].values - N_fit_eo * ho_df["Et"].values
        C_fit, r2_water = fit_zero_intercept(pi_minus_p, f_residual)
        st.plotly_chart(
            plot_water_drive_diagnostic(pi_minus_p, f_residual, C_fit, r2_water),
            width='stretch',
        )
        st.markdown(
            "**Equation:** (F - N * Et) = C * (Pi - P)\n\n"
            "Points scatter randomly around zero → No significant water influx. "
            "Depletion or gas cap drive is dominant.\n\n"
            "Points trend upward with increasing (Pi − P) → Active water influx. "
            "Slope gives aquifer constant C. Note: steady-state aquifer assumed. "
            "Large aquifers require Van Everdingen-Hurst for rigorous analysis."
        )

        # Drive Mechanism Summary Card
        st.subheader("Drive Mechanism Summary")
        # Rewired to pull exclusively from the NNLS joint fit (fit_joint_drive_parameters,
        # already computed in drive_params before st.tabs()) rather than the sequential
        # 1D fits above -- those assume m=0 for the depletion estimate and m=best_m (which
        # can itself be a poor fit) for the gas-cap estimate, which is why the old logic
        # could show >500% OOIP inflation on combination-drive data. Primary mechanism is
        # now the argmax of the three drive indices (compute_drive_indices) at the final
        # step, not sequential R² comparison.
        N_est, m_est, C_est, r2_joint = drive_params
        final_indices = indices_df.dropna().iloc[-1] if not indices_df.dropna().empty else None

        if final_indices is not None:
            mechanism_shares = {
                "Depletion": final_indices["DDI"],
                "Gas Cap": final_indices["SDI"],
                "Water Drive": final_indices["WDI"],
            }
            mechanism = max(mechanism_shares, key=mechanism_shares.get)
        else:
            mechanism = "Depletion"

        water_evidence = C_est > 1e-6
        n_points = len(production_df)

        if r2_joint > 0.95 and n_points >= 15 and not warnings_only:
            confidence = "High"
        elif r2_joint < 0.85 or n_points < 8:
            confidence = "Low"
        else:
            confidence = "Moderate"

        st.markdown(
            f"""
```
DRIVE MECHANISM DIAGNOSIS — SUMMARY
─────────────────────────────────────────────────────
  Primary mechanism:    {mechanism}
  Estimated OOIP:       {N_est / 1e6:.1f} MMstb
  Gas cap ratio (m):    {m_est:.2f}
  R² (joint NNLS fit):  {r2_joint:.3f}
  Aquifer constant (C): {C_est:,.0f} rb/psia
  Aquifer evidence:     {"Yes" if water_evidence else "No"}
─────────────────────────────────────────────────────
  CONFIDENCE: {confidence}
```
            """
        )
        st.caption(
            "Negative R² means this drive mechanism model fits worse than a flat "
            "line. Try a different diagnostic tab."
        )
        # Ground Truth Validation (comparing against a known true_params) is
        # gone along with the Synthetic Data Generator UI -- there is no
        # ground truth for real uploaded data. src/synthetic.py's true_params
        # output still exists and is exercised by tests/test_synthetic.py.

# --- TAB 4: Insights & Drive Indices ----------------------------------------
with tab4:
    if not data_available:
        st.info("Upload production and PVT data in Tab 1 first.")
    elif blocking:
        st.error("Blocking validation errors — see banners below. Analysis cannot proceed.")
        _render_validation_banners()
    else:
        if warnings_only:
            _render_validation_banners()

        st.subheader("Drive Index Summary")
        N_est, m_est, C_est, r2_joint = drive_params
        last_valid = indices_df.dropna()

        if last_valid.empty:
            st.info("Not enough production history yet to compute drive indices.")
        else:
            final_row = last_valid.iloc[-1]
            ddi_pct = final_row["DDI"] * 100
            sdi_pct = final_row["SDI"] * 100
            wdi_pct = final_row["WDI"] * 100
            index_total = ddi_pct + sdi_pct + wdi_pct

            col1, col2, col3 = st.columns(3)
            col1.metric("DDI — Depletion Drive", f"{ddi_pct:.1f}%")
            col2.metric("SDI — Gas Cap Drive", f"{sdi_pct:.1f}%")
            col3.metric("WDI — Water Drive", f"{wdi_pct:.1f}%")
            st.caption(
                f"Sum = {index_total:.1f}% at the final recorded time step "
                f"(joint N/m/C fit R² = {r2_joint:.3f}; small deviation from "
                "100% reflects the joint fit's residual, not a calculation error)."
            )

            st.plotly_chart(plot_drive_indices(ho_df, indices_df), width='stretch')

            mechanisms_pct = {"Depletion": ddi_pct, "Gas Cap": sdi_pct, "Water Influx": wdi_pct}
            ranked = sorted(mechanisms_pct.items(), key=lambda kv: kv[1], reverse=True)
            primary_label, primary_pct = ranked[0]
            secondary_label, secondary_pct = ranked[1]
            st.info(
                f"At the final recorded time step, this reservoir is primarily "
                f"driven by {primary_label} ({primary_pct:.0f}%), followed by "
                f"{secondary_label} ({secondary_pct:.0f}%)."
            )

# Action 2: once Tabs 2, 3, and 4 have all shown their error banners for a
# blocking validation error, stop the script -- nothing meaningful follows
# this point (the sidebar already rendered earlier in the script).
if data_available and blocking:
    st.stop()
