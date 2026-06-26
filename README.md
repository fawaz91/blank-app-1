# SurvExtrapolate AI Trial Suite

A Streamlit prototype for clinical survival digitization, parametric extrapolation, treatment-effect adjustment, registry validation, and health-economic partitioned survival modelling.

## Features

- Trial login scaffold with GDPR-oriented privacy controls and local trial-session access.
- Kaplan-Meier figure upload workflow with automatic digitization after a user selects or highlights a line.
- At-risk table capture with treatment-arm mapping.
- Parametric survival fitting for exponential, Weibull, log-normal, log-logistic, gamma-like, Gompertz, and weighted-model workflows.
- Relative-risk, background mortality, and constant hazard increase/decrease adjustments.
- Standard-error estimates, AIC/BIC comparison, follow-up uncertainty controls, and optional registry comparison.
- Built-in AI troubleshooting assistant for calibration, digitization, at-risk, and extrapolation problems.
- Partitioned survival economic modelling with progression-free, post-progression, death, discontinuation, QALY, disutility, cost, and optional Markov-state extension controls.
- Excel, TreeAge-ready CSV, and R script export.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
