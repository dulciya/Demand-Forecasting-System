import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def run_random_forest(product_name="Laptop", forecast_only=False):

    # Load dataset
    df = pd.read_csv("data/retail_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    product_df = df[df["Product"] == product_name].copy()

    # Create time-based features
    product_df["Month"] = product_df["Date"].dt.month
    product_df["Year"] = product_df["Date"].dt.year
    product_df["Time_Index"] = np.arange(len(product_df))

    X = product_df[["Month", "Year", "Time_Index"]]
    y = product_df["Units_Sold"]

    # Train-test split
    X_train = X[:-6]
    X_test = X[-6:]
    y_train = y[:-6]
    y_test = y[-6:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    if forecast_only:
        return predictions

    return rmse