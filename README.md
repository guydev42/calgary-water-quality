# Water quality anomaly detection

## Problem statement
Calgary's watersheds supply drinking water to 1.4 million residents. Manual review of thousands of lab results across dozens of monitoring sites is impractical. This project builds an ensemble anomaly detection system that flags unusual water quality measurements across the Bow and Elbow river networks, enabling faster response to contamination events.

## Approach
- Fetched watershed water quality data (82 parameters, 7 monitoring sites) from Calgary Open Data
- Pivoted long-form measurements into wide-form, one column per parameter
- Engineered rolling statistics, rate-of-change, and z-score features per site
- Applied four detection methods: Isolation Forest, Local Outlier Factor, 3-sigma rule, and z-score
- Combined all four into a majority-vote ensemble for balanced precision and recall
- Built a Streamlit dashboard with site-level anomaly maps and parameter drill-downs

## Key results

| Method | Precision | Recall | F1 score |
|--------|-----------|--------|----------|
| **Ensemble (majority vote)** | **~0.75** | **~0.73** | **~0.74** |
| Isolation Forest | ~0.72 | ~0.68 | ~0.70 |
| Local Outlier Factor | ~0.65 | ~0.74 | ~0.69 |
| Statistical (3-sigma) | ~0.80 | ~0.55 | ~0.65 |

## How to run
```bash
pip install -r requirements.txt
python src/data_loader.py    # fetch water quality data
streamlit run app.py         # launch dashboard
```

## Project structure
```
project_10_water_quality_anomaly_detection/
├── app.py                  # Streamlit dashboard
├── requirements.txt
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

## Technical stack
pandas, NumPy, scikit-learn (Isolation Forest, LOF), Plotly, Streamlit, sodapy, joblib
