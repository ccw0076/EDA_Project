# College Basketball EDA Dashboard

**Authors: Carly Walker and Richard Wilbanks**

This repository contains a full exploratory data analysis (EDA) of NCAA Division I men's basketball statistics from 2010–2025, along with an interactive Streamlit dashboard that visualizes trends, correlations, outliers, conference comparisons, and team performance insights.

### Project Overview

This project explores:

Which statistics correlate most strongly with winning?

How offensive/defensive metrics shifted over time (2010–2025)

SEC team “statistical signatures”

Outlier teams by NET (point differential)

Power conferences (SEC, ACC, Big 10, Big 12, Big East)

Auburn vs. National Champions comparison

Custom calculated stats (PPG, OPPG, NET, Avg Point Differential)


### Includes:

Correlation plots

Time-series visualizations

Heatmaps

Boxplots

Scatterplots with regression lines

Radar charts comparing Auburn to national champions

### Python EDA Notebook

A companion Jupyter notebook explores:

Data cleaning

Outlier detection

Visualizations

Statistical summaries

### Dependencies Used

Python 3.10+

Pandas

NumPy

Plotly

Streamlit

SciPy

JupyterLab

### How to Run the Dashboard Locally
1. Install Dependencies
pip install streamlit pandas plotly numpy scipy

2. Launch the Streamlit App
streamlit run basketball_dashboard.py


The dashboard will automatically open in your browser.

### Data Source

Data collected from sports-reference.com/cbb

Includes all Division I teams from 2010–2025
