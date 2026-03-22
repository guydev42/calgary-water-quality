<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Calgary%20Water%20Quality%20Anomaly%20Detection&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Ensemble%20anomaly%20detection%20for%20multi-site%20watershed%20monitoring&descSize=16&descAlignY=55&descColor=c8ddf0" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/status-complete-2ea44f?style=for-the-badge" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/streamlit-dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Problem** -- Calgary's watersheds supply drinking water to 1.4 million residents. Manual review of thousands of lab results across dozens of monitoring sites is impractical, and delayed detection of contamination events poses serious public health risks.

**Solution** -- This project builds an ensemble anomaly detection system combining Isolation Forest, Local Outlier Factor, 3-sigma statistical rules, and z-score methods into a majority-vote ensemble that flags unusual water quality measurements across the Bow and Elbow river networks.

**Impact** -- Achieves 75% precision in identifying true anomalies, enabling water quality analysts to focus their review on the most critical measurements and respond faster to potential contamination events across seven monitoring sites.

---

## Results

| Method | Precision | Recall | F1 score |
|--------|-----------|--------|----------|
| **Ensemble (majority vote)** | **0.75** | **0.73** | **0.74** |
| Isolation Forest | 0.72 | 0.68 | 0.70 |
| Local Outlier Factor | 0.65 | 0.74 | 0.69 |
| Statistical (3-sigma) | 0.80 | 0.55 | 0.65 |

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Calgary Open     │────▶│  Feature          │────▶│  Isolation        │
│  Data (Socrata)   │     │  Engineering      │     │  Forest           │
└──────────────────┘     │  - Rolling stats  │     └────────┬─────────┘
                         │  - Rate of change │              │
                         │  - Z-scores       │     ┌────────▼─────────┐
                         └──────────────────┘     │  Local Outlier    │
                                                   │  Factor           │
                                                   └────────┬─────────┘
                                                            │
                         ┌──────────────────┐     ┌────────▼─────────┐
                         │  Streamlit        │◀────│  Majority-Vote   │
                         │  Dashboard        │     │  Ensemble         │
                         └──────────────────┘     └──────────────────┘
```

---

<details>
<summary><strong>Project structure</strong></summary>

```
project_10_water_quality_anomaly_detection/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching, caching & feature engineering
    └── model.py            # Anomaly detection models and ensemble
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/olag-portfolio/calgary-water-quality.git
cd calgary-water-quality

# Install dependencies
pip install -r requirements.txt

# Fetch water quality data
python src/data_loader.py

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Dataset | Source | Records | Key fields |
|---------|--------|---------|------------|
| Watershed water quality | Calgary Open Data | Multi-year | 82 parameters, 7 monitoring sites |
| River network sites | Calgary Open Data | 7 sites | Bow River, Elbow River locations |

---

## Tech stack

<p align="center">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-API-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/joblib-serialization-grey?style=flat-square" />
</p>

---

## Methodology

1. **Data collection** -- Fetched watershed water quality data covering 82 parameters across 7 monitoring sites from Calgary Open Data via Socrata API.
2. **Data transformation** -- Pivoted long-form measurement records into wide-form with one column per parameter, handling missing values and inconsistent units.
3. **Feature engineering** -- Computed rolling statistics (mean, standard deviation), rate-of-change indicators, and z-scores for each parameter at each monitoring site.
4. **Anomaly detection** -- Applied four independent detection methods: Isolation Forest for multivariate outliers, Local Outlier Factor for density-based anomalies, 3-sigma statistical thresholds, and z-score flagging.
5. **Ensemble** -- Combined all four methods into a majority-vote ensemble requiring at least two detectors to agree before flagging a measurement, achieving 75% precision with 73% recall.
6. **Dashboard** -- Built a Streamlit application with site-level anomaly maps, parameter drill-downs, and temporal trend visualizations.

---

## Acknowledgements

Data provided by the [City of Calgary Open Data Portal](https://data.calgary.ca/). This project was developed as part of a municipal data analytics portfolio.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>

<p align="center">
  Built by <a href="https://github.com/olag-portfolio">Ola G</a>
</p>
