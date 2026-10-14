"""Builds one clean CSV for Tableau from the two raw CSVs your project already saved in artifacts/."""
import pandas as pd
from pathlib import Path

art = Path("artifacts")
risk = pd.read_csv(art / "risk_raw.csv")
freight = pd.read_csv(art / "freight_raw.csv", parse_dates=["InvoiceDate", "PODate", "PayDate"])

# tidy vendor names (they have trailing spaces in the source)
risk["VendorName"] = risk["VendorName"].str.strip()
freight["VendorName"] = freight["VendorName"].str.strip()

# join the two tables on the purchase order number
cols = ["PONumber", "VendorNumber", "InvoiceDate", "PODate", "PayDate", "Approval"]
df = risk.merge(freight[cols], on="PONumber", how="left", validate="one_to_one")

# helpful columns for the dashboard
df["risk_label"] = df["flag_invoice"].map({1: "Risky", 0: "OK"})
df["invoice_month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
df["dollar_mismatch"] = (df["invoice_dollars"] - df["total_item_dollars"]).round(2)
df["quantity_mismatch"] = df["invoice_quantity"] - df["total_item_quantity"]
df["freight_pct_of_invoice"] = (100 * df["Freight"] / df["invoice_dollars"]).round(2)

out = Path("tableau"); out.mkdir(exist_ok=True)
df.to_csv(out / "invoice_dashboard_data.csv", index=False)

print("rows:", len(df), "| missing InvoiceDate after join:", df["InvoiceDate"].isna().sum())
print("risky invoices:", int(df["flag_invoice"].sum()), f"({100 * df['flag_invoice'].mean():.1f}%)")
print("vendors:", df["VendorName"].nunique(), "| months:", df["invoice_month"].min().date(), "to", df["invoice_month"].max().date())
print("wrote tableau/invoice_dashboard_data.csv")
