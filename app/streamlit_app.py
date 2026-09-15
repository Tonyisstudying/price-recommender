from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

API_URL = os.getenv("PRICING_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="SEA Smart Pricing", layout="wide")
st.title("SEA E-commerce Intelligent Pricing Engine")
st.caption("Cost + target margin + competitor distribution + ML + demand optimization")

with st.sidebar:
    st.header("Product")
    country = st.selectbox("Country", ["Vietnam", "Thailand", "Malaysia", "Singapore", "Indonesia", "Philippines"])
    marketplace = st.selectbox("Marketplace", ["Shopee", "Lazada"])
    category = st.selectbox("Category", ["Accessories", "Groceries", "Beauty", "Electronics", "Home & Living"])
    product_name = st.text_input("Product name", "Logitech Wireless Mouse")
    brand = st.text_input("Brand", "Logitech")
    cost = st.number_input("Unit cost (USD)", min_value=0.01, value=8.00, step=0.10)
    margin = st.slider("Target gross margin", min_value=0.05, max_value=0.80, value=0.30, step=0.01)
    strategy = st.selectbox("Pricing strategy", ["competitive", "balanced", "premium"], index=1)
    run = st.button("Recommend price", type="primary", use_container_width=True)


def call_api():
    response = requests.post(
        f"{API_URL}/api/v1/pricing/recommend",
        json={
            "country": country,
            "marketplace": marketplace,
            "category": category,
            "product_name": product_name,
            "brand": brand,
            "cost": cost,
            "target_margin": margin,
            "strategy": strategy,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


if run:
    try:
        st.session_state["result"] = call_api()
    except requests.RequestException as exc:
        st.error(f"Could not connect to pricing API: {exc}")

result = st.session_state.get("result")

if result:
    st.subheader(f"Recommended price: ${result['recommended_price']:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Margin floor", f"${result['minimum_profitable_price']:.2f}")
    c2.metric("Competitor median", f"${result['competitor_median']:.2f}")
    c3.metric("ML market price", f"${result['predicted_market_price']:.2f}")
    c4.metric("Optimized price", f"${result['optional_optimized_price']:.2f}")

    if result.get("warning"):
        st.warning(result["warning"])
    else:
        st.success("Target margin is compatible with the current market range.")

    st.subheader("Competitive distribution")
    distribution = pd.DataFrame({
        "Metric": ["P25", "Median", "P75"],
        "Price": [result["competitor_p25"], result["competitor_median"], result["competitor_p75"]],
    })
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.plot(distribution["Price"], [1] * len(distribution), "o-")
    for _, row in distribution.iterrows():
        ax.annotate(f"{row['Metric']}: ${row['Price']:.2f}", (row["Price"], 1), xytext=(0, 12), textcoords="offset points", ha="center")
    ax.axvline(result["recommended_price"], linestyle="--", label="Recommended")
    ax.set_yticks([])
    ax.set_xlabel("USD")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Explanation")
    for reason in result["explanation"]:
        st.write(f"- {reason}")

    st.subheader("Comparable products")
    st.dataframe(pd.DataFrame(result["comparable_products"]), use_container_width=True, hide_index=True)

    st.subheader("Demand-aware optimization")
    st.write(f"Expected demand: {result['expected_demand']:.2f} units")
    st.write(f"Expected contribution profit: ${result['expected_profit']:.2f}")
else:
    st.info("Enter product information and click 'Recommend price'.")
