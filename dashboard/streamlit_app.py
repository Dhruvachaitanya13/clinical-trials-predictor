"""
Day 5: Clinical Trials Landscape — a professionally designed analytics dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Design system
-------------
- Palette and typography live in .streamlit/config.toml (deep-teal clinical theme).
- A small CSS layer adds card surfaces, KPI styling, and spacing rhythm.
- Information architecture is tabbed: Overview | Drivers | Trends | Explore.
- Charts share one color sequence and layout template for visual consistency.
- Every analytical view pairs a chart with a written takeaway (the analyst skill).

Engineering
-----------
- Cached data loading.
- Each section is fault-isolated: one broken chart never blanks the page.
- Graceful empty/missing-data handling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants & design tokens
# --------------------------------------------------------------------------- #
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "trials_features.csv"
COMPLETED = "COMPLETED"

PHASE_ORDER = [
    "EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2",
    "PHASE2", "PHASE2/PHASE3", "PHASE3", "PHASE4",
]

# Cohesive teal/slate sequence used across every chart
SEQ = ["#0E7C7B", "#3FA7A6", "#7FC9C8", "#C7A03C", "#B5651D", "#5B6C6C", "#9AA8A8"]
TEAL = "#0E7C7B"
SLATE = "#1A2B2B"
GRID = "#E2E8E8"

st.set_page_config(
    page_title="Clinical Trials Landscape",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# CSS layer — restrained, purposeful (cards, KPIs, spacing, headings)
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      /* KPI cards */
      .kpi {
        background: #FFFFFF; border: 1px solid #E2E8E8; border-radius: 0.9rem;
        padding: 1.1rem 1.25rem; box-shadow: 0 1px 2px rgba(16,40,40,0.04);
      }
      .kpi .label { font-size: 0.78rem; color: #5B6C6C; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-bottom: 0.35rem; }
      .kpi .value { font-size: 1.7rem; font-weight: 700; color: #0E2A2A; line-height: 1.1; }
      .kpi .sub   { font-size: 0.8rem; color: #7C8A8A; margin-top: 0.2rem; }
      /* Section intro text */
      .section-note { color: #5B6C6C; font-size: 0.95rem; margin: -0.4rem 0 0.8rem 0; }
      /* Finding callout */
      .finding {
        background: #F1F5F5; border-left: 3px solid #0E7C7B; border-radius: 0.4rem;
        padding: 0.75rem 1rem; color: #2A3B3B; font-size: 0.92rem; margin-top: 0.4rem;
      }
      /* Tighter divider */
      hr { margin: 1.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading trial data…")
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run the Day 1 notebook to generate it."
        )
    df = pd.read_csv(path)
    df["completed"] = (df["overall_status"].str.upper() == COMPLETED).astype(int)
    if "phase" in df:
        df["phase"] = df["phase"].fillna("UNKNOWN").str.upper()
    if "sponsor_class" in df:
        df["sponsor_class"] = df["sponsor_class"].fillna("UNKNOWN")
    if "start_year" in df:
        df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
    return df


def style_fig(fig: go.Figure, *, pct_axis: str | None = None) -> go.Figure:
    """Apply one consistent layout template to every chart."""
    fig.update_layout(
        template="simple_white",
        colorway=SEQ,
        font=dict(family="sans-serif", color=SLATE, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if pct_axis == "y":
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
    elif pct_axis == "x":
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return fig


# --------------------------------------------------------------------------- #
# Reusable UI atoms
# --------------------------------------------------------------------------- #
def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f"<div class='kpi'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def finding(text: str) -> None:
    st.markdown(f"<div class='finding'>{text}</div>", unsafe_allow_html=True)


class section:
    """Fault-isolated section context manager."""
    def __init__(self, title: str, note: str = ""):
        self.title, self.note = title, note

    def __enter__(self):
        st.markdown(f"### {self.title}")
        if self.note:
            st.markdown(f"<div class='section-note'>{self.note}</div>", unsafe_allow_html=True)
        return self

    def __exit__(self, et, ev, tb):
        if et is not None:
            st.warning(f"Could not render '{self.title}': {ev}")
            return True
        return False


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def fig_phase(df):
    g = (df.groupby("phase")["completed"].agg(["mean", "count"]).reset_index()
         .rename(columns={"mean": "rate", "count": "n"}))
    g["phase"] = pd.Categorical(g["phase"], categories=PHASE_ORDER, ordered=True)
    g = g.dropna(subset=["phase"]).sort_values("phase")
    fig = px.bar(g, x="phase", y="rate", text="n")
    fig.update_traces(marker_color=TEAL, texttemplate="%{text:,}", textposition="outside")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Completion rate")
    return style_fig(fig, pct_axis="y")


def fig_sponsor(df):
    g = (df.groupby("sponsor_class")["completed"].agg(["mean", "count"]).reset_index()
         .rename(columns={"mean": "rate", "count": "n"}).sort_values("rate"))
    fig = px.bar(g, x="rate", y="sponsor_class", orientation="h", text="n",
                 color="rate", color_continuous_scale=["#7FC9C8", "#0E7C7B"])
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Completion rate")
    fig.update_yaxes(title="")
    return style_fig(fig, pct_axis="x")


def fig_enrollment(df):
    d = df[df["enrollment"].notna() & (df["enrollment"] >= 0)].copy()
    cap = d["enrollment"].quantile(0.99)
    d["bucket"] = pd.qcut(d["enrollment"].clip(upper=cap), q=6, duplicates="drop")
    g = d.groupby("bucket", observed=True)["completed"].mean().reset_index()
    g["bucket"] = g["bucket"].astype(str)
    fig = px.bar(g, x="bucket", y="completed")
    fig.update_traces(marker_color=TEAL)
    fig.update_xaxes(title="Enrollment range", tickangle=-25)
    fig.update_yaxes(title="Completion rate")
    return style_fig(fig, pct_axis="y")


def fig_trend(df):
    d = df[df["start_year"].between(2005, 2022)]
    g = d.groupby("start_year")["completed"].mean().reset_index()
    fig = px.line(g, x="start_year", y="completed", markers=True)
    fig.update_traces(line_color=TEAL, marker_color=TEAL)
    fig.update_xaxes(title="Start year")
    fig.update_yaxes(title="Completion rate")
    return style_fig(fig, pct_axis="y")


def fig_volume(df):
    d = df[df["start_year"].between(2005, 2022)]
    g = d.groupby(["start_year", "completed"]).size().reset_index(name="n")
    g["Outcome"] = g["completed"].map({1: "Completed", 0: "Not completed"})
    fig = px.area(g, x="start_year", y="n", color="Outcome",
                  color_discrete_map={"Completed": TEAL, "Not completed": "#C7A03C"})
    fig.update_xaxes(title="Start year")
    fig.update_yaxes(title="Trials")
    return style_fig(fig)


def fig_phase_sponsor(df):
    d = df[df["phase"].isin(PHASE_ORDER)]
    g = d.pivot_table(index="phase", columns="sponsor_class",
                      values="completed", aggfunc="mean")
    g = g.reindex([p for p in PHASE_ORDER if p in g.index])
    fig = go.Figure(data=go.Heatmap(
        z=g.values, x=list(g.columns), y=list(g.index),
        colorscale=["#F1F5F5", "#0E7C7B"], zmin=0, zmax=1,
        colorbar=dict(title="Rate", tickformat=".0%"),
    ))
    return style_fig(fig)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## 🧪 Trials Explorer")
    st.sidebar.caption("AACT / ClinicalTrials.gov")
    st.sidebar.divider()
    st.sidebar.markdown("#### Filters")

    phases = sorted(df["phase"].dropna().unique())
    pick_p = st.sidebar.multiselect("Phase", phases, default=phases)
    sponsors = sorted(df["sponsor_class"].dropna().unique())
    pick_s = st.sidebar.multiselect("Sponsor class", sponsors, default=sponsors)

    if df["start_year"].notna().any():
        ymin, ymax = int(df["start_year"].min()), int(df["start_year"].max())
        yr = st.sidebar.slider("Start year", ymin, ymax,
                               (max(ymin, 2005), min(ymax, 2022)))
    else:
        yr = None

    out = df[df["phase"].isin(pick_p) & df["sponsor_class"].isin(pick_s)]
    if yr:
        out = out[out["start_year"].between(yr[0], yr[1]) | out["start_year"].isna()]

    st.sidebar.divider()
    st.sidebar.metric("Trials in view", f"{len(out):,}")
    return out


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        df_all = load_data(DATA_PATH)
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()

    df = sidebar(df_all)

    st.title("Clinical Trials Landscape")
    st.markdown(
        "<div class='section-note'>Where do interventional trials complete — and where "
        "do they fail? Patterns across phase, sponsor, enrollment, and time, derived from "
        "the AACT mirror of ClinicalTrials.gov.</div>",
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("No trials match the current filters."); st.stop()

    # KPI band
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Trials", f"{len(df):,}", "after filters")
    kpi(c2, "Completion rate", f"{df['completed'].mean():.1%}", "completed vs. not")
    if "enrollment" in df:
        kpi(c3, "Median enrollment", f"{int(df['enrollment'].median()):,}", "participants")
    if df["start_year"].notna().any():
        kpi(c4, "Year span", f"{int(df['start_year'].min())}–{int(df['start_year'].max())}", "start dates")

    st.divider()

    tab_over, tab_drivers, tab_trends, tab_explore = st.tabs(
        ["Overview", "Drivers", "Trends", "Explore"]
    )

    with tab_over:
        with section("Completion rate by phase",
                     "Bar labels show the number of trials in each phase."):
            st.plotly_chart(fig_phase(df), use_container_width=True)
            finding("<b>Finding:</b> Completion is fairly high across phases (73–85%). Phase 1 and Phase 3 lead "
                    "at ~85%, while combination phases (Phase 1/Phase 2 ~73%, Phase 2 ~77%) trail — the "
                    "transitional stages, where efficacy is still unproven, are the most likely to stop early.")

    with tab_drivers:
        a, b = st.columns(2)
        with a:
            with section("By sponsor type"):
                st.plotly_chart(fig_sponsor(df), use_container_width=True)
                finding("<b>Finding:</b> Completion is remarkably even across funded sponsor types — industry, NIH, and "
                        "federal all sit near 84%. The differences between established sponsors are small; funding "
                        "source matters less here than trial design and size.")
        with b:
            with section("By enrollment size"):
                st.plotly_chart(fig_enrollment(df), use_container_width=True)
                finding("<b>Finding:</b> The strongest signal in the data: very small trials (≤16 participants) complete only "
                        "~47% of the time, while every larger bucket exceeds 87% and the largest reach ~93%. "
                        "Tiny, likely exploratory or under-powered studies are far more prone to early termination.")
        with section("Phase × sponsor interaction",
                     "Heatmap of completion rate; darker = higher."):
            st.plotly_chart(fig_phase_sponsor(df), use_container_width=True)
            finding("<b>Finding:</b> Across well-populated cells, late-phase industry and federal trials are the most "
                    "reliable, while early/transitional phases run lower regardless of sponsor. (A sparse ‘AMBIG’ "
                    "sponsor category produces unstable 0%/NaN cells and is excluded from interpretation.)")

    with tab_trends:
        with section("Completion rate over time"):
            st.plotly_chart(fig_trend(df), use_container_width=True)
            finding("<b>Finding:</b> Recorded completion drifts down from ~83% (2008) to ~74% (2020). Much of this "
                    "decline is a data artifact: more recent trials are still in progress or not yet resolved, so they "
                    "are under-counted as completed — not evidence that trials are genuinely failing more often.")
        with section("Trial volume over time, by outcome"):
            st.plotly_chart(fig_volume(df), use_container_width=True)
            finding("<b>Finding:</b> Trial volume rises steadily across the period. The completed-vs-not mix is stable "
                    "in earlier years but skews toward not-yet-resolved outcomes more recently — the same labeling "
                    "artifact visible in the trend above.")

    with tab_explore:
        with section("Filtered records", "First 200 rows of the current selection."):
            cols = [c for c in ["nct_id", "phase", "sponsor_class", "enrollment",
                                "overall_status", "start_year"] if c in df.columns]
            st.dataframe(df[cols].head(200), use_container_width=True, height=420)
            st.download_button(
                "⬇ Download filtered data (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_trials.csv", mime="text/csv",
            )

    st.divider()
    st.caption("Built with Streamlit · Data: AACT / ClinicalTrials.gov · "
               "Completion = trial reached its planned end (not a measure of clinical efficacy).")


if __name__ == "__main__":
    main()
