from src.forecasting import run_arima_forecast
from src.evaluation import evaluate_arima
from random_forest_model import run_random_forest
from src.inventory import calculate_inventory

print("=== Demand Forecasting & Inventory Optimization System ===")

product = "Laptop"

# Evaluate ARIMA
arima_rmse, arima_mape = evaluate_arima(product)
print(f"\nARIMA RMSE: {round(arima_rmse,2)}")
print(f"ARIMA MAPE: {round(arima_mape,2)}%")

# Evaluate Random Forest
rf_rmse = run_random_forest(product)
print(f"\nRandom Forest RMSE: {round(rf_rmse,2)}")

# Model Selection
if arima_rmse < rf_rmse:
    print("\nBest Model Selected: ARIMA")
else:
    print("\nBest Model Selected: Random Forest")

# Inventory
inventory_results = calculate_inventory(product)

print("\nInventory Analysis")
print("------------------")
for key, value in inventory_results.items():
    print(f"{key}: {round(value,2) if isinstance(value,float) else value}")