import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def run_arima_forecast(product_name="Laptop", steps=6):

    df = pd.read_csv("data/retail_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    product_df = df[df["Product"] == product_name]
    product_df.set_index("Date", inplace=True)

    sales = product_df["Units_Sold"]

    model = ARIMA(sales, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    return sales, forecast