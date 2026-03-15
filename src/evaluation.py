import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def evaluate_arima(product_name="Laptop"):

    df = pd.read_csv("data/retail_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    product_df = df[df["Product"] == product_name]
    product_df.set_index("Date", inplace=True)

    sales = product_df["Units_Sold"]

    train = sales[:-6]
    test = sales[-6:]

    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=6)

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    return rmse, mape