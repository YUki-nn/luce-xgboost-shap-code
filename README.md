# LUCE XGBoost-SHAP Code Repository

This repository contains the code and supporting materials used for the machine learning analysis in the manuscript:

"An integrated framework for identifying nonlinear drivers and peaking trajectories of land-use carbon emissions: A case study of island cities"

## Overview

The repository provides scripts for:

- Data preprocessing
- XGBoost model training
- Optuna hyperparameter optimization
- SHAP interpretation
- Feature importance analysis
- Nonlinear response analysis
- Threshold detection
- PDP and ICE analysis
- Figure generation

## Repository structure

```text
scripts/
  01_Model_Training.py

requirements.txt
README.md
```

## Main methods

The workflow includes:

- XGBoost regression
- SHAP analysis
- SHAP dependence plots
- LOWESS smoothing
- Piecewise threshold detection
- Partial dependence analysis (PDP)
- Individual conditional expectation (ICE)
- Cross-validation
- Hyperparameter optimization using Optuna

## Software environment

Python 3.10+

Main packages:

- xgboost
- shap
- optuna
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels

## Data availability

Some raw datasets used in this study are subject to data-use restrictions and therefore cannot be publicly redistributed.

This repository provides the code and supporting materials necessary to reproduce the machine learning workflow and main analytical procedures described in the manuscript.

## Contact

For questions regarding the code or analysis, please contact:

gwf101@163.com
