# Demand Forecasting & Inventory Optimization System

This project predicts product demand and automatically recommends inventory reorder decisions using machine learning and time-series forecasting.

## Features
- Demand forecasting using ARIMA
- Demand prediction using Random Forest
- Model comparison using RMSE and MAPE
- Automatic best model selection
- Inventory optimization using safety stock and reorder point
- End-to-end forecasting pipeline

## Tech Stack
Python  
Pandas  
NumPy  
Scikit-learn  
Statsmodels  
Matplotlib

## Project Structure
data/ – retail sales dataset  
src/ – forecasting, evaluation, and inventory modules  
run_pipeline.py – runs the full forecasting system  
output/ – generated forecast plots

## Example Output
ARIMA RMSE: 13.03  
ARIMA MAPE: 5.2%  
Random Forest RMSE: 12.76  

Best Model Selected: Random Forest

Inventory Decision:
Reorder Required

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the pipeline:

python run_pipeline.py