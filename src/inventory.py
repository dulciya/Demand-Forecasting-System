import pandas as pd
import numpy as np

def calculate_inventory(product_name="Laptop", service_level=1.65, lead_time=2):

    # Load dataset
    df = pd.read_csv("data/retail_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter product
    product_df = df[df["Product"] == product_name]

    sales = product_df["Units_Sold"]

    avg_demand = sales.mean()
    std_demand = sales.std()

    # Safety Stock formula
    safety_stock = service_level * std_demand * np.sqrt(lead_time)

    # Reorder Point formula
    reorder_point = (avg_demand * lead_time) + safety_stock

    current_inventory = 150  # assumed value

    if current_inventory < reorder_point:
        decision = "Reorder Required"
        reorder_qty = reorder_point - current_inventory
    else:
        decision = "Stock Sufficient"
        reorder_qty = 0

    return {
        "Product": product_name,
        "Average Monthly Demand": avg_demand,
        "Demand Std Dev": std_demand,
        "Safety Stock": safety_stock,
        "Reorder Point": reorder_point,
        "Current Inventory": current_inventory,
        "Decision": decision,
        "Suggested Reorder Quantity": reorder_qty
    }