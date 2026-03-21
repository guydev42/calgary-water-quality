"""
Calgary Water Quality Anomaly Detection System -- Streamlit Application.

Launch with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure the project src package is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    fetch_water_quality_data,
    preprocess,
    pivot_parameters,
    add_rolling_statistics,
    add_rate_of_change,
    add_zscore_features,
    KEY_PARAMETERS,
)
from src.model import WaterQualityAnomalyDetector, detect_anomalies

# ======================================================================
# Page config
# ======================================================================
st.set_page_config(
    page_title="Calgary Water Quality Anomaly Detection System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Cached data loading
# ======================================================================

@st.cache_data(show_spinner="Loading water quality data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch, preprocess, pivot, and feature-engineer the dataset."""
    raw = fetch_water_quality_data(limit=50_000)
    long_df = preprocess(raw)
    wide_df = pivot_parameters(long_df)
    wide_df = add_rolling_statistics(wide_df)
    wide_df = add_rate_of_change(wide_df)
    wide_df = add_zscore_features(wide_df)
    return long_df, wide_df


@st.cache_resource(show_spinner="Running anomaly detection...")
def run_detection(
    _wide_df: pd.DataFrame,
    feature_cols: tuple,
    contamination: float,
    threshold: float,
):
    """Run the ensemble anomaly detector (cached as a resource)."""
    detector, results, summary = detect_anomalies(
        _wide_df,
        list(feature_cols),
        contamination=contamination,
        ensemble_threshold=threshold,
    )
    return detector, results, summary


# ======================================================================
# Helper utilities
# ======================================================================

def _get_parameter_columns(wide_df: pd.DataFrame) -> list[str]:
    """Return parameter column names from the wide dataframe."""
    exclude = {
        "sample_site", "sample_date", "year", "month",
    }
    return [
        c for c in wide_df.columns
        if c not in exclude
        and "_roll_" not in c
        and "_roc" not in c
        and "_zscore" not in c
        and wide_df[c].dtype in ("float64", "int64")
    ]


def _get_feature_columns(wide_df: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns (including engineered)."""
    exclude = {"sample_site", "sample_date", "year", "month"}
    return [
        c for c in wide_df.columns
        if c not in exclude and wide_df[c].dtype in ("float64", "int64")
    ]


# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Water Quality Dashboard",
        "Anomaly Detection",
        "Site Analysis",
        "Parameter Correlations",
        "About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Calgary Water Quality Anomaly Detection System")

# ======================================================================
# Data loading (shared across pages)
# ======================================================================
try:
    long_df, wide_df = load_data()
    data_loaded = True
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    data_loaded = False

# ======================================================================
# Pages
# ======================================================================

# ------------------------------------------------------------------
# 1. Water Quality Dashboard
# ------------------------------------------------------------------
if page == "Water Quality Dashboard" and data_loaded:
    st.title("Water Quality Dashboard")
    st.markdown("Overview of watershed water quality monitoring across Calgary.")

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monitoring Sites", long_df["sample_site"].nunique())
    col2.metric("Total Samples", f"{len(long_df):,}")
    col3.metric("Parameters Monitored", long_df["parameter"].nunique())
    date_range = f"{long_df['sample_date'].min().date()} to {long_df['sample_date'].max().date()}"
    col4.metric("Date Range", date_range)

    st.markdown("---")

    # Site selector for time series
    sites = sorted(long_df["sample_site"].dropna().unique())
    selected_site = st.selectbox("Select Monitoring Site", sites, key="dash_site")

    # Determine which key parameters exist in the data
    available_params = long_df["parameter"].unique()
    plot_params = [p for p in KEY_PARAMETERS if p in available_params]
    if not plot_params:
        plot_params = list(available_params[:5])

    # Time series of key parameters
    st.subheader("Time Series of Key Parameters")
    site_data = long_df[
        (long_df["sample_site"] == selected_site) & (long_df["parameter"].isin(plot_params))
    ].dropna(subset=["numeric_result"])

    if not site_data.empty:
        fig_ts = px.line(
            site_data,
            x="sample_date",
            y="numeric_result",
            color="parameter",
            title=f"Key Parameters Over Time -- {selected_site}",
            labels={"sample_date": "Date", "numeric_result": "Value", "parameter": "Parameter"},
        )
        fig_ts.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No data available for the selected site and key parameters.")

    # Parameter distribution boxplots
    st.subheader("Parameter Value Distributions")
    param_for_box = st.selectbox("Select Parameter", sorted(available_params), key="dash_box_param")
    box_data = long_df[long_df["parameter"] == param_for_box].dropna(subset=["numeric_result"])

    if not box_data.empty:
        fig_box = px.box(
            box_data,
            x="sample_site",
            y="numeric_result",
            title=f"Distribution of {param_for_box} by Site",
            labels={"sample_site": "Site", "numeric_result": param_for_box},
        )
        fig_box.update_layout(height=450, template="plotly_white", xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No data available for the selected parameter.")

# ------------------------------------------------------------------
# 2. Anomaly Detection
# ------------------------------------------------------------------
elif page == "Anomaly Detection" and data_loaded:
    st.title("Anomaly Detection")
    st.markdown("Run multi-method ensemble anomaly detection on water quality data.")

    param_cols = _get_parameter_columns(wide_df)
    feat_cols = _get_feature_columns(wide_df)

    # Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        sites = sorted(wide_df["sample_site"].dropna().unique())
        selected_site = st.selectbox("Select Site", ["All Sites"] + list(sites), key="ad_site")
    with col_ctrl2:
        selected_param = st.selectbox("Focus Parameter", ["All"] + param_cols, key="ad_param")

    col_ctrl3, col_ctrl4 = st.columns(2)
    with col_ctrl3:
        contamination = st.slider("Contamination (expected anomaly %)", 1, 20, 5, key="ad_cont") / 100.0
    with col_ctrl4:
        ensemble_threshold = st.slider("Ensemble Threshold", 0.25, 1.0, 0.5, 0.05, key="ad_thresh")

    # Filter data
    analysis_df = wide_df.copy()
    if selected_site != "All Sites":
        analysis_df = analysis_df[analysis_df["sample_site"] == selected_site].copy()

    if analysis_df.empty or len(analysis_df) < 5:
        st.warning("Not enough data for the selected filters. Please adjust your selection.")
    else:
        run_btn = st.button("Run Anomaly Detection", type="primary")
        if run_btn:
            use_feats = tuple(feat_cols)
            detector, results, summary = run_detection(
                analysis_df, use_feats, contamination, ensemble_threshold,
            )

            analysis_df = analysis_df.copy()
            analysis_df["anomaly"] = results["anomaly"]
            analysis_df["ensemble_score"] = results["ensemble_score"]

            n_anomalies = int(results["anomaly"].sum())
            st.success(f"Detection complete -- **{n_anomalies}** anomalies found out of {len(analysis_df)} samples.")

            # Time series with anomalies highlighted
            st.subheader("Anomalies on Time Series")
            if selected_param != "All" and selected_param in analysis_df.columns:
                plot_param = selected_param
            else:
                plot_param = param_cols[0] if param_cols else None

            if plot_param and plot_param in analysis_df.columns:
                fig_anom = go.Figure()
                normal = analysis_df[analysis_df["anomaly"] == 0]
                anom = analysis_df[analysis_df["anomaly"] == 1]

                fig_anom.add_trace(go.Scatter(
                    x=normal["sample_date"], y=normal[plot_param],
                    mode="markers+lines", name="Normal",
                    marker=dict(color="steelblue", size=5),
                    line=dict(color="steelblue", width=1),
                ))
                fig_anom.add_trace(go.Scatter(
                    x=anom["sample_date"], y=anom[plot_param],
                    mode="markers", name="Anomaly",
                    marker=dict(color="red", size=10, symbol="x"),
                ))
                site_label = selected_site if selected_site != "All Sites" else "All Sites"
                fig_anom.update_layout(
                    title=f"{plot_param} with Detected Anomalies -- {site_label}",
                    xaxis_title="Date",
                    yaxis_title=plot_param,
                    template="plotly_white",
                    height=450,
                )
                st.plotly_chart(fig_anom, use_container_width=True)

            # Anomaly summary table
            st.subheader("Anomaly Summary Table")
            if not summary.empty:
                st.dataframe(summary, use_container_width=True, height=350)
            else:
                st.info("No anomalies detected with current settings.")

            # Detection method comparison
            st.subheader("Detection Method Comparison")
            method_counts = {
                "Isolation Forest": int(results["isolation_forest"].sum()),
                "Local Outlier Factor": int(results["lof"].sum()),
                "Statistical (3-sigma)": int(results["statistical"].sum()),
                "Z-Score": int(results["zscore"].sum()),
                "Ensemble": n_anomalies,
            }
            comp_df = pd.DataFrame(
                list(method_counts.items()),
                columns=["Method", "Anomalies Detected"],
            )
            fig_comp = px.bar(
                comp_df, x="Method", y="Anomalies Detected",
                title="Anomalies Detected per Method",
                color="Method",
            )
            fig_comp.update_layout(template="plotly_white", showlegend=False, height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------------------------------------------------
# 3. Site Analysis
# ------------------------------------------------------------------
elif page == "Site Analysis" and data_loaded:
    st.title("Site Analysis")
    st.markdown("Deep-dive into individual monitoring sites.")

    sites = sorted(long_df["sample_site"].dropna().unique())
    selected_site = st.selectbox("Select Monitoring Site", sites, key="sa_site")
    site_long = long_df[long_df["sample_site"] == selected_site]

    # All parameters over time
    st.subheader(f"All Parameters Over Time -- {selected_site}")
    if not site_long.empty:
        fig_all = px.line(
            site_long.dropna(subset=["numeric_result"]),
            x="sample_date",
            y="numeric_result",
            color="parameter",
            title=f"Parameters Over Time -- {selected_site}",
            labels={"sample_date": "Date", "numeric_result": "Value", "parameter": "Parameter"},
        )
        fig_all.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_all, use_container_width=True)

    # Site-level statistics
    st.subheader("Site-Level Statistics")
    stats = (
        site_long.dropna(subset=["numeric_result"])
        .groupby("parameter")["numeric_result"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(3)
        .reset_index()
    )
    stats.columns = ["Parameter", "Count", "Mean", "Std Dev", "Min", "Median", "Max"]
    st.dataframe(stats, use_container_width=True)

    # Map of monitoring sites
    st.subheader("Monitoring Site Locations")
    coord_df = (
        long_df.dropna(subset=["latitude_degrees", "longitude_degrees"])
        .groupby("sample_site")[["latitude_degrees", "longitude_degrees"]]
        .first()
        .reset_index()
    )
    if not coord_df.empty:
        coord_df["is_selected"] = coord_df["sample_site"] == selected_site
        fig_map = px.scatter_mapbox(
            coord_df,
            lat="latitude_degrees",
            lon="longitude_degrees",
            hover_name="sample_site",
            color="is_selected",
            color_discrete_map={True: "red", False: "blue"},
            zoom=9,
            height=500,
            title="Monitoring Sites (selected site in red)",
        )
        fig_map.update_layout(mapbox_style="open-street-map", showlegend=False)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No coordinate data available for mapping.")

# ------------------------------------------------------------------
# 4. Parameter Correlations
# ------------------------------------------------------------------
elif page == "Parameter Correlations" and data_loaded:
    st.title("Parameter Correlations")
    st.markdown("Explore relationships and seasonal patterns among water quality parameters.")

    param_cols = _get_parameter_columns(wide_df)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    if len(param_cols) >= 2:
        corr = wide_df[param_cols].corr()
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Parameter Correlation Matrix",
            aspect="auto",
        )
        fig_heat.update_layout(height=550, template="plotly_white")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough numeric parameters for a correlation matrix.")

    # Scatter plots of parameter pairs
    st.subheader("Parameter Pair Scatter Plot")
    if len(param_cols) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            param_x = st.selectbox("X-axis Parameter", param_cols, index=0, key="corr_x")
        with col_b:
            default_y = 1 if len(param_cols) > 1 else 0
            param_y = st.selectbox("Y-axis Parameter", param_cols, index=default_y, key="corr_y")

        scatter_df = wide_df.dropna(subset=[param_x, param_y])
        if not scatter_df.empty:
            fig_sc = px.scatter(
                scatter_df,
                x=param_x,
                y=param_y,
                color="sample_site",
                title=f"{param_x} vs {param_y}",
                opacity=0.6,
            )
            fig_sc.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_sc, use_container_width=True)

    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    season_param = st.selectbox("Select Parameter for Seasonal View", param_cols, key="season_param")
    if season_param in long_df["parameter"].values:
        season_data = long_df[long_df["parameter"] == season_param].dropna(subset=["numeric_result"])
    elif season_param in wide_df.columns:
        season_data = wide_df[["sample_date", season_param]].dropna().copy()
        season_data["month"] = season_data["sample_date"].dt.month
        season_data.rename(columns={season_param: "numeric_result"}, inplace=True)
    else:
        season_data = pd.DataFrame()

    if not season_data.empty and "month" in season_data.columns:
        fig_season = px.box(
            season_data,
            x="month",
            y="numeric_result",
            title=f"Seasonal Pattern of {season_param}",
            labels={"month": "Month", "numeric_result": season_param},
        )
        fig_season.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.info("Insufficient data for seasonal analysis of this parameter.")

# ------------------------------------------------------------------
# 5. About
# ------------------------------------------------------------------
elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Calgary Water Quality Anomaly Detection System

    ### Why Watershed Water Quality Matters

    Calgary's watersheds -- the Bow and Elbow river systems -- supply drinking
    water to over 1.4 million residents and support critical aquatic
    ecosystems.  Continuous monitoring of water quality parameters such as
    **pH**, **temperature**, **dissolved oxygen**, **turbidity**, and
    **conductance** is essential for:

    - **Early warning** of contamination events (industrial spills, sewage
      overflows, agricultural runoff).
    - **Regulatory compliance** with provincial and federal water quality
      standards.
    - **Ecological health** assessment of aquatic habitats.
    - **Long-term trend analysis** to guide infrastructure and policy
      decisions.

    ### Dataset

    This application uses the **Watershed Water Quality** dataset published by
    the City of Calgary on its Open Data Portal (dataset ID: `y8as-bmzj`).
    The data includes laboratory results from multiple monitoring sites across
    the watershed, spanning parameters like pH, temperature, conductance,
    dissolved oxygen, and many more.

    ### Methodology: Multi-Method Ensemble Anomaly Detection

    Rather than relying on a single algorithm, this system combines **four
    complementary anomaly detection methods** into an ensemble:

    | Method | How It Works |
    |--------|-------------|
    | **Isolation Forest** | Builds random decision trees; anomalies are isolated in fewer splits. Effective for high-dimensional data. |
    | **Local Outlier Factor (LOF)** | Measures local density deviation; points in sparser regions are flagged. Captures cluster-based anomalies. |
    | **Statistical (3-sigma)** | Flags values beyond three standard deviations from the mean. Simple and interpretable baseline. |
    | **Z-Score** | Computes per-feature z-scores; flags samples where any feature exceeds the threshold. Complements the global statistical method. |

    The **ensemble score** is the mean of the four binary predictions
    (0 = normal, 1 = anomaly).  A configurable threshold (default 0.5 --
    majority vote) determines the final classification.

    ### Feature Engineering

    - **Rolling statistics** (7-day and 30-day mean and standard deviation)
      capture recent trends.
    - **Rate-of-change** (first difference) highlights sudden shifts.
    - **Z-score features** provide normalised context for each measurement.

    ### How to Run

    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

    ### Technology Stack

    - **Python 3.10+**
    - **Streamlit** -- interactive web dashboard
    - **Plotly** -- interactive visualisations
    - **scikit-learn** -- Isolation Forest & LOF
    - **pandas / NumPy** -- data wrangling
    - **sodapy** -- Socrata Open Data API client
    """)

    st.markdown("---")
    st.caption("Calgary Water Quality Anomaly Detection System | Data sourced from Calgary Open Data Portal")
