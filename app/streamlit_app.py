import streamlit as st

from src.pricing import (
    recommend_price_v2,
)


st.set_page_config(
    page_title="SEA Smart Pricing",
    layout="wide"
)

st.title(
    "SEA E-commerce Smart Pricing Engine"
)

st.write(
    "Cost + target gross margin + competitor "
    "price distribution + ML market estimate"
)

st.sidebar.header("Product")

country = st.sidebar.selectbox(
    "Country",
    [
        "Vietnam",
        "Thailand",
        "Malaysia",
        "Singapore",
        "Indonesia",
        "Philippines",
    ]
)

category = st.sidebar.selectbox(
    "Category",
    [
        "Accessories",
        "Groceries",
        "Beauty",
        "Electronics",
        "Home & Living",
    ]
)

cost = st.sidebar.number_input(
    "Unit cost",
    min_value=0.01,
    value=8.00,
)

margin = st.sidebar.slider(
    "Target gross margin",
    min_value=0.05,
    max_value=0.80,
    value=0.30,
    step=0.01,
)

strategy = st.sidebar.selectbox(
    "Pricing strategy",
    [
        "competitive",
        "balanced",
        "premium",
    ]
)

if st.button("Recommend Price"):

    # Placeholder competitor stats
    # Replace these with actual matcher results.
    competitor_stats = {
        "p25": 10.80,
        "median": 11.20,
        "p75": 12.10,
    }

    # Replace with actual ML prediction.
    predicted_market_price = 11.50

    recommended = recommend_price_v2(
        cost=cost,
        target_margin=margin,
        stats=competitor_stats,
        predicted_market_price=
            predicted_market_price,
        strategy=strategy,
    )

    st.subheader(
        f"Recommended price: ${recommended:.2f}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Minimum profitable price",
            f"${cost / (1 - margin):.2f}"
        )

    with col2:
        st.metric(
            "Competitor median",
            f"${competitor_stats['median']:.2f}"
        )

    with col3:
        st.metric(
            "Recommended price",
            f"${recommended:.2f}"
        )

    st.subheader(
        "Competitive price distribution"
    )

    st.write(
        f"""
        P25: ${competitor_stats['p25']:.2f}

        Median: ${competitor_stats['median']:.2f}

        P75: ${competitor_stats['p75']:.2f}
        """
    )