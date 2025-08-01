import pandas as pd
from pathlib import Path

INPUT = "data/ticket_prices.csv"
OUTPUT = "data/analysis_summary.csv"

def main():
    df = pd.read_csv(INPUT, parse_dates=["date"])
    group_cols = ["theatre", "seat_type"]

    avg_prices = (
        df.groupby(group_cols + ["platform"])["total_price"]
        .mean()
        .reset_index()
        .pivot_table(index=group_cols, columns="platform", values="total_price")
        .reset_index()
    )

    def cheaper_platform(row):
        if pd.isna(row.get("BMS")) or pd.isna(row.get("Paytm")):
            return None
        if row["BMS"] < row["Paytm"]:
            return "BMS"
        elif row["Paytm"] < row["BMS"]:
            return "Paytm"
        return "Tie"

    def price_diff(row):
        if pd.isna(row.get("BMS")) or pd.isna(row.get("Paytm")):
            return None
        return abs(row["BMS"] - row["Paytm"])

    avg_prices["cheaper_platform"] = avg_prices.apply(cheaper_platform, axis=1)
    avg_prices["price_diff"] = avg_prices.apply(price_diff, axis=1)

    avg_prices.to_csv(OUTPUT, index=False)
    print(f"Saved summary to {OUTPUT}")

if __name__ == "__main__":
    main()
