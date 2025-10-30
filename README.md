# House Price Prediction in Madrid

This repository contains my **Bachelor’s Thesis (TFG)** focused on **predicting real estate prices in Madrid** using machine learning and deep learning.  
The workflow can be adapted to other Spanish cities such as **Barcelona** or **Valencia** using Idealista data.

## Project Overview

The project combines **data collection**, **feature engineering**, and **predictive modeling**.  
Main steps:

1. **Data acquisition** – download `.rda` files from [Idealista18](https://github.com/paezha/idealista18) and convert to `.csv`.  
2. **Feature creation** – add distances to hospitals, metro stations, and other points of interest.  
3. **Exploratory analysis** – study correlations and remove irrelevant features.  
4. **Modeling** – train **XGBoost** and **TensorFlow** models on raw and processed datasets.  
5. **Evaluation** – compare predictions with real apartment prices in Madrid.

## Folder Structure

1.0 Data download & conversion
2.0 Points of interest
4.0 Basic statistics
5.0 Grid-based data
6.0 XGBoost models
7.0 TensorFlow models
8.0 Visualizations
9.0 Real apartment tests


## Requirements

Python ≥ 3.9  
Main libraries: pandas, numpy, matplotlib, xgboost, tensorflow, rdata, scikit-learn


## Author

**Pablo López** – Bachelor’s Thesis, Universidad de Navarra
