"""All Plotly figure builders (spec v6, Section 6, Tab 2 and Tab 3 plots).

Pure rendering layer: every function takes already-computed data (production
DataFrames, H-O term arrays, regression results) and returns a
plotly.graph_objects.Figure. No physics or regression is computed here --
that belongs to material_balance.py / synthetic.py.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "oil":          "#2E8B57",   # green
    "gas":          "#DC143C",   # red
    "water":        "#1E90FF",   # blue
    "pressure":     "#FF8C00",   # orange
    "fitted_line":  "#333333",   # dark grey
    "data_points":  "#6A0DAD",   # purple
    "bubble_point": "#FFD700",   # gold — for Pb reference line
}

FONT_FAMILY = "Arial, sans-serif"
FONT_SIZE_AXIS = 13
FONT_SIZE_TITLE = 15

GRIDLINE_KWARGS = dict(showgrid=True, gridcolor="rgba(128,128,128,0.3)")


def _safe_ratio(numerator, denominator):
    """Elementwise numerator/denominator, NaN wherever denominator is ~0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(denominator) > 1e-12, numerator / denominator, np.nan)
    return ratio


def _apply_layout_defaults(fig, title, xaxis_title, yaxis_title, showlegend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE)),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
        showlegend=showlegend,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=xaxis_title, **GRIDLINE_KWARGS)
    fig.update_yaxes(title_text=yaxis_title, **GRIDLINE_KWARGS)
    return fig


# ---------------------------------------------------------------------------
# TAB 2: Production Diagnostics
# ---------------------------------------------------------------------------

def plot_production_history(production_df):
    """
    Plot 1 — Production History.
    Left Y-axis: oil rate (STB/day), from the oil_rates_noisy column directly
    (NOT delta-Np/delta-t — see spec Section 5, Tab 2, Plot 1).
    Right Y-axis: GOR and WOR per step.
    """
    dates = production_df["Date"]
    oil_rate = production_df["oil_rates_noisy"].to_numpy(dtype=float)
    gas_rate = production_df["gas_rates_noisy"].to_numpy(dtype=float)
    water_rate = production_df["water_rates_noisy"].to_numpy(dtype=float)
    gor = _safe_ratio(gas_rate, oil_rate)
    wor = _safe_ratio(water_rate, oil_rate)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=dates, y=oil_rate, name="Oil rate (STB/day)", mode="lines",
        line=dict(color=COLORS["oil"]),
        hovertemplate="Date: %{x}<br>Oil rate: %{y:,.0f} STB/day<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=gor, name="GOR (Scf/STB)", mode="lines",
        line=dict(color=COLORS["gas"], dash="dot"),
        hovertemplate="Date: %{x}<br>GOR: %{y:,.0f} Scf/STB<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=dates, y=wor, name="WOR (STB/STB)", mode="lines",
        line=dict(color=COLORS["water"], dash="dash"),
        hovertemplate="Date: %{x}<br>WOR: %{y:.3f} STB/STB<extra></extra>",
    ), secondary_y=True)

    if len(oil_rate) > 0 and np.any(~np.isnan(oil_rate)):
        peak_idx = int(np.nanargmax(oil_rate))
        fig.add_annotation(
            x=dates.iloc[peak_idx], y=oil_rate[peak_idx],
            text="Peak rate", showarrow=True, arrowhead=2, yref="y",
        )

    fig.update_xaxes(title_text="Date", **GRIDLINE_KWARGS)
    fig.update_yaxes(title_text="Oil rate (STB/day)", secondary_y=False, **GRIDLINE_KWARGS)
    fig.update_yaxes(title_text="GOR (Scf/STB) / WOR (STB/STB)", secondary_y=True, **GRIDLINE_KWARGS)
    fig.update_layout(
        title=dict(text="Production History", font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE)),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
        showlegend=True,
        hovermode="x unified",
    )
    return fig


def plot_gor_vs_np(production_df):
    """Plot 2 — GOR vs cumulative oil (Np)."""
    Np_mm = production_df["Np"].to_numpy(dtype=float) / 1e6
    oil_rate = production_df["oil_rates_noisy"].to_numpy(dtype=float)
    gas_rate = production_df["gas_rates_noisy"].to_numpy(dtype=float)
    gor = _safe_ratio(gas_rate, oil_rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Np_mm, y=gor, mode="lines+markers", name="GOR",
        line=dict(color=COLORS["gas"]), marker=dict(color=COLORS["gas"], size=5),
        hovertemplate="Np: %{x:.3f} MMstb<br>GOR: %{y:,.0f} Scf/STB<extra></extra>",
    ))
    return _apply_layout_defaults(
        fig, "GOR vs Cumulative Oil", "Np (MMstb)", "GOR (Scf/STB)", showlegend=False
    )


def plot_wor_vs_np(production_df):
    """Plot 3 — WOR vs cumulative oil (Np), semi-log Y-axis."""
    Np_mm = production_df["Np"].to_numpy(dtype=float) / 1e6
    oil_rate = production_df["oil_rates_noisy"].to_numpy(dtype=float)
    water_rate = production_df["water_rates_noisy"].to_numpy(dtype=float)
    wor = _safe_ratio(water_rate, oil_rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Np_mm, y=wor, mode="lines+markers", name="WOR",
        line=dict(color=COLORS["water"]), marker=dict(color=COLORS["water"], size=5),
        hovertemplate="Np: %{x:.3f} MMstb<br>WOR: %{y:.4f} STB/STB<extra></extra>",
    ))
    fig = _apply_layout_defaults(
        fig, "WOR vs Cumulative Oil (Semi-log)", "Np (MMstb)", "WOR (STB/STB, log scale)",
        showlegend=False,
    )
    fig.update_yaxes(type="log")
    return fig


def plot_pressure_vs_np(production_df, Pb):
    """Plot 4 — Reservoir pressure vs cumulative oil (Np), with a Pb reference line."""
    Np_mm = production_df["Np"].to_numpy(dtype=float) / 1e6
    P = production_df["P_avg"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Np_mm, y=P, mode="lines+markers", name="Reservoir pressure",
        line=dict(color=COLORS["pressure"]), marker=dict(color=COLORS["pressure"], size=5),
        hovertemplate="Np: %{x:.3f} MMstb<br>P: %{y:,.0f} psia<extra></extra>",
    ))
    fig.add_hline(
        y=Pb, line=dict(color=COLORS["bubble_point"], dash="dash"),
        annotation_text="Bubble Point", annotation_position="top left",
    )

    # Flag the exact point where P first drops below Pb (saturated region begins).
    below_pb = np.where(P < Pb)[0]
    if len(below_pb) > 0:
        cross_idx = int(below_pb[0])
        fig.add_trace(go.Scatter(
            x=[Np_mm[cross_idx]], y=[P[cross_idx]], mode="markers",
            name="Bubble Point Reached",
            marker=dict(color="red", size=14, symbol="circle",
                        line=dict(color="black", width=1)),
            hovertemplate="Bubble Point Reached<br>Np: %{x:.3f} MMstb<br>P: %{y:,.0f} psia<extra></extra>",
        ))
        fig.add_annotation(
            x=Np_mm[cross_idx], y=P[cross_idx], text="Bubble Point Reached",
            showarrow=True, arrowhead=2, ax=40, ay=-40,
        )

    return _apply_layout_defaults(
        fig, "Reservoir Pressure vs Cumulative Oil", "Np (MMstb)", "Average reservoir pressure (psia)",
        showlegend=True,
    )


# ---------------------------------------------------------------------------
# TAB 3: Havlena-Odeh Analysis
# ---------------------------------------------------------------------------

def plot_f_vs_eo(Eo, F, N_fit, r2):
    """
    Section 3A — F vs Eo (Depletion Diagnosis).
    Scatter + zero-intercept fitted line, with a residuals subplot below.
    F is expected in rb; plotted in MMrb.
    """
    Eo = np.asarray(Eo, dtype=float)
    F_mm = np.asarray(F, dtype=float) / 1e6
    fitted_mm = (N_fit * Eo) / 1e6
    residuals_mm = F_mm - fitted_mm
    steps = np.arange(len(Eo))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, row_heights=[0.7, 0.3],
        vertical_spacing=0.12,
        subplot_titles=("F vs Eo (Depletion Diagnosis)", "Residuals (F actual − F fitted)"),
    )
    fig.add_trace(go.Scatter(
        x=Eo, y=F_mm, mode="markers", name="Data", marker=dict(color=COLORS["data_points"], size=7),
        hovertemplate="Eo: %{x:.4f} rb/STB<br>F: %{y:.3f} MMrb<extra></extra>",
    ), row=1, col=1)
    order = np.argsort(Eo)
    fig.add_trace(go.Scatter(
        x=Eo[order], y=fitted_mm[order], mode="lines", name="Zero-intercept fit",
        line=dict(color=COLORS["fitted_line"]),
    ), row=1, col=1)
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.02, y=0.95, showarrow=False,
        text=f"Estimated OOIP = {N_fit / 1e6:.1f} MMstb | R² = {r2:.3f}",
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(
        x=steps, y=residuals_mm, mode="markers", name="Residuals",
        marker=dict(color=COLORS["data_points"], size=6), showlegend=False,
        hovertemplate="Step: %{x}<br>Residual: %{y:.4f} MMrb<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=COLORS["fitted_line"], dash="dot"), row=2, col=1)

    fig.update_xaxes(title_text="Eo (rb/STB)", row=1, col=1, **GRIDLINE_KWARGS)
    fig.update_yaxes(title_text="F (MMrb)", row=1, col=1, **GRIDLINE_KWARGS)
    fig.update_xaxes(title_text="Time step", row=2, col=1, **GRIDLINE_KWARGS)
    fig.update_yaxes(title_text="Residual (MMrb)", row=2, col=1, **GRIDLINE_KWARGS)
    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
        title=dict(text="F vs Eo (Depletion Diagnosis)", font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE)),
        showlegend=True,
    )
    return fig


def plot_f_vs_et(Et, F, N_fit, r2, m):
    """
    Section 3B — F vs Et (Gas Cap Diagnosis).
    Et = Eo + m*Eg + Efw is computed by the caller (it depends on the live
    m slider) and passed in already-combined.
    """
    Et = np.asarray(Et, dtype=float)
    F_mm = np.asarray(F, dtype=float) / 1e6
    fitted_mm = (N_fit * Et) / 1e6

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Et, y=F_mm, mode="markers", name="Data", marker=dict(color=COLORS["data_points"], size=7),
        hovertemplate="Et: %{x:.4f} rb/STB<br>F: %{y:.3f} MMrb<extra></extra>",
    ))
    order = np.argsort(Et)
    fig.add_trace(go.Scatter(
        x=Et[order], y=fitted_mm[order], mode="lines", name="Zero-intercept fit",
        line=dict(color=COLORS["fitted_line"]),
    ))
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.02, y=0.95, showarrow=False,
        text=f"m = {m:.2f} | Estimated OOIP = {N_fit / 1e6:.1f} MMstb | R² = {r2:.3f}",
    )
    return _apply_layout_defaults(
        fig, "F vs Et (Gas Cap Diagnosis)", "Et = Eo + m·Eg + Efw (rb/STB)", "F (MMrb)"
    )


def plot_water_drive_diagnostic(pi_minus_p, f_residual, C_fit, r2):
    """
    Section 3C — Water Drive Diagnosis.
    X-axis: (Pi - P) in psia. Y-axis: (F - N*Et) in MMrb, where N is the
    best depletion fit from Section 3A. Slope of the zero-intercept fit is
    the aquifer constant C (rb/psia). Do NOT plot against time step index.
    """
    pi_minus_p = np.asarray(pi_minus_p, dtype=float)
    f_residual_mm = np.asarray(f_residual, dtype=float) / 1e6
    fitted_mm = (C_fit * pi_minus_p) / 1e6

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pi_minus_p, y=f_residual_mm, mode="markers", name="Data",
        marker=dict(color=COLORS["data_points"], size=7),
        hovertemplate="Pi-P: %{x:,.0f} psia<br>F-N·Et: %{y:.3f} MMrb<extra></extra>",
    ))
    order = np.argsort(pi_minus_p)
    fig.add_trace(go.Scatter(
        x=pi_minus_p[order], y=fitted_mm[order], mode="lines", name="Zero-intercept fit",
        line=dict(color=COLORS["fitted_line"]),
    ))
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.02, y=0.95, showarrow=False,
        text=f"Aquifer constant C = {C_fit:,.0f} rb/psia | R² = {r2:.3f}",
    )
    return _apply_layout_defaults(
        fig, "Water Drive Diagnosis", "Pi - P (psia)", "F - N·Et (MMrb)"
    )


# ---------------------------------------------------------------------------
# TAB 4: Insights & Drive Indices
# ---------------------------------------------------------------------------

def plot_drive_indices(ho_df, indices_df):
    """
    Renders DDI, SDI, WDI as a 100% stacked area chart over time (row/time
    step index, since ho_df carries no date column). Each row is
    renormalized to sum to exactly 100% for display, so the chart always
    reads as a true 100% stacked area regardless of small H-O fit residuals
    in the underlying (raw, physically meaningful) indices_df.
    """
    steps = np.arange(len(indices_df))
    raw = indices_df[["DDI", "SDI", "WDI"]].to_numpy(dtype=float)
    row_sum = np.nansum(raw, axis=1)
    row_sum = np.where(row_sum > 1e-9, row_sum, np.nan)
    normalized = raw / row_sum[:, None] * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=normalized[:, 0], mode="lines", name="DDI (Depletion)",
        stackgroup="one", line=dict(color=COLORS["oil"]),
        hovertemplate="Step: %{x}<br>DDI: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=normalized[:, 1], mode="lines", name="SDI (Gas Cap)",
        stackgroup="one", line=dict(color=COLORS["gas"]),
        hovertemplate="Step: %{x}<br>SDI: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=normalized[:, 2], mode="lines", name="WDI (Water Drive)",
        stackgroup="one", line=dict(color=COLORS["water"]),
        hovertemplate="Step: %{x}<br>WDI: %{y:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 100])
    return _apply_layout_defaults(
        fig, "Drive Index Contribution Over Time", "Time step", "Share of total withdrawal (%)"
    )
