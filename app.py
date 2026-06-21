import sys
import numpy as np
import pandas as pd
import streamlit as st

from src.exception import CustomException
from src.utils import load_object


# =========================================================================
# Load all artifacts once (Streamlit caches this so it doesn't reload
# on every user interaction)
# =========================================================================

@st.cache_resource
def load_freight_artifacts():
    model = load_object("artifacts/freight_model.pkl")
    preprocessor = load_object("artifacts/freight_preprocessor.pkl")
    vendor_map = load_object("artifacts/freight_vendor_map.pkl")
    return model, preprocessor, vendor_map


@st.cache_resource
def load_risk_artifacts():
    model = load_object("artifacts/risk_model.pkl")
    preprocessor = load_object("artifacts/risk_preprocessor.pkl")
    vendor_map = load_object("artifacts/risk_vendor_map.pkl")
    return model, preprocessor, vendor_map


def encode_vendor(vendor_name, vendor_map):
    """Look up the vendor's target-encoded value, falling back to the
    overall training mean if it's a vendor the model has never seen."""
    means = vendor_map["vendor_means"]
    overall_mean = vendor_map["overall_mean"]
    return means.get(vendor_name, overall_mean)


st.set_page_config(page_title="Invoice Intelligence System", layout="centered")
st.title("📦 Invoice Intelligence System")
st.write("Predict freight cost and assess invoice risk for vendor purchase orders.")

tab1, tab2 = st.tabs(["💰 Freight Cost Prediction", "⚠️ Invoice Risk Assessment"])

# =========================================================================
# TAB 1: FREIGHT COST PREDICTION
# =========================================================================
with tab1:
    st.header("Freight Cost Prediction")
    st.write("Enter order details to estimate the expected freight cost.")

    try:
        freight_model, freight_preprocessor, freight_vendor_map = load_freight_artifacts()
        known_vendors = sorted(freight_vendor_map["vendor_means"].keys())

        col1, col2 = st.columns(2)
        with col1:
            vendor_name = st.selectbox("Vendor Name", known_vendors, key="freight_vendor")
            quantity = st.number_input("Quantity", min_value=1, value=100, key="freight_qty")
            dollars = st.number_input("Order Value (Dollars)", min_value=1.0, value=5000.0, key="freight_dollars")

        with col2:
            days_po_to_invoice = st.number_input("Days: PO to Invoice", min_value=0, value=16, key="freight_d1")
            days_invoice_to_pay = st.number_input("Days: Invoice to Pay", min_value=0, value=35, key="freight_d2")
            days_po_to_pay = st.number_input("Days: PO to Pay (total)", min_value=0, value=52, key="freight_d3")

        if st.button("Predict Freight Cost", type="primary"):
            input_df = pd.DataFrame({
                "Quantity_log": [np.log1p(quantity)],
                "Dollars_log": [np.log1p(dollars)],
                "days_po_to_invoice": [days_po_to_invoice],
                "days_invoice_to_pay": [days_invoice_to_pay],
                "days_po_to_pay": [days_po_to_pay],
                "VendorName_encoded": [encode_vendor(vendor_name, freight_vendor_map)]
            })

            transformed = freight_preprocessor.transform(input_df)
            prediction = freight_model.predict(transformed)[0]
            prediction = max(0, prediction)  # freight can't be negative

            st.success(f"### Estimated Freight Cost: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Error loading freight model: {e}")

# =========================================================================
# TAB 2: INVOICE RISK ASSESSMENT
# =========================================================================
with tab2:
    st.header("Invoice Risk Assessment")
    st.write("Enter invoice and purchase details to assess whether this invoice should be flagged for review.")

    try:
        risk_model, risk_preprocessor, risk_vendor_map = load_risk_artifacts()
        known_risk_vendors = sorted(risk_vendor_map["vendor_means"].keys())

        col1, col2 = st.columns(2)
        with col1:
            r_vendor_name = st.selectbox("Vendor Name", known_risk_vendors, key="risk_vendor")
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=100, key="risk_qty")
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=5000.0, key="risk_dollars")
            freight = st.number_input("Freight Cost", min_value=0.0, value=25.0, key="risk_freight")
            days_po_to_invoice_r = st.number_input("Days: PO to Invoice", min_value=0, value=16, key="risk_d1")

        with col2:
            days_to_pay = st.number_input("Days: Invoice to Pay", min_value=0, value=35, key="risk_d2")
            total_brands = st.number_input("Total Distinct Brands on PO", min_value=1, value=3, key="risk_brands")
            total_item_quantity = st.number_input("Total Item Quantity (from Purchases)", min_value=1, value=100, key="risk_item_qty")
            total_item_dollars = st.number_input("Total Item Dollars (from Purchases)", min_value=1.0, value=5000.0, key="risk_item_dollars")
            avg_receiving_delay = st.number_input("Avg. Receiving Delay (days)", min_value=0.0, value=8.0, key="risk_delay")

        if st.button("Assess Invoice Risk", type="primary"):
            input_df = pd.DataFrame({
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "days_po_to_invoice": [days_po_to_invoice_r],
                "days_to_pay": [days_to_pay],
                "total_brands": [total_brands],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars],
                "avg_receiving_delay": [avg_receiving_delay],
                "VendorName_encoded": [encode_vendor(r_vendor_name, risk_vendor_map)]
            })

            transformed = risk_preprocessor.transform(input_df)
            prediction = risk_model.predict(transformed)[0]

            if prediction == 1:
                st.error("### ⚠️ This invoice is flagged as RISKY")
                st.write("Recommend manual review before processing.")
            else:
                st.success("### ✅ This invoice appears LOW RISK")
                st.write("No red flags detected based on current data.")

    except Exception as e:
        st.error(f"Error loading risk model: {e}")