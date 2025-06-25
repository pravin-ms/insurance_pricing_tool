
# insurance_pricing_tool_mulberri_brand.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Mulberri AI Insurance Pricing Tool")

# Mulberri Theme Styling
st.markdown("""
<style>
    .main { background-color: #f7fafd; }
    h1, h2, h3, h4 { color: #2e1a47; }
    .stButton button { background-color: #2e1a47; color: white; }
    .sidebar .sidebar-content { background-color: #2e1a47; color: white; }
    .status-green { color: green; font-weight: bold; }
    .status-orange { color: orange; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.image("https://cdn.prod.website-files.com/67b608ea58e266b5f621042c/67b63d2d6b6941fdc36b31cd_logo-mulberri.svg", width=170)

st.title("Mulberri AI-Powered Insurance Pricing Tool")

col1, col2 = st.columns(2)
with col1:
    quotes_file = st.file_uploader("📥 Upload Quotes CSV", type="csv")
with col2:
    claims_file = st.file_uploader("📥 Upload Claims CSV", type="csv")

if quotes_file and claims_file:
    quotes_df = pd.read_csv(quotes_file)
    claims_df = pd.read_csv(claims_file)

    quotes_df['Result'] = quotes_df['Result'].map({'Win': 1, 'Loss': 0})
    quotes_df['Normalized Risk Amount'] = quotes_df['Normalized Risk Amount'].fillna(0)
    quotes_df['Relative Price'] = quotes_df['PriceQuoted'] / quotes_df['Normalized Risk Amount']

    quotes_encoded = pd.get_dummies(quotes_df, columns=['Country', 'Sector', 'Broker'])
    X = quotes_encoded.drop(['QuoteID', 'Result', 'PriceQuoted'], axis=1)
    X['Relative Price'] = quotes_df['Relative Price'].values
    y = quotes_encoded['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    status_color = "status-green" if accuracy >= 0.75 else "status-orange"
    st.markdown(f"<div class='{status_color}'>✅ Model trained with {accuracy:.2%} accuracy</div>", unsafe_allow_html=True)

    st.header("🎯 AI Pricing Recommendation")

    with st.form("recommendation_table_form"):
        country = st.selectbox("Country", sorted(quotes_df['Country'].unique()))
        sector = st.selectbox("Sector", sorted(quotes_df['Sector'].unique()))
        broker = st.selectbox("Broker", sorted(quotes_df['Broker'].unique()))
        base_price = st.number_input("📌 Base Quoted Price ($)", min_value=0.0, value=5000.0)
        risk_amount = st.number_input("📊 Normalized Risk Amount ($)", min_value=0.0, value=10000.0)
        submitted = st.form_submit_button("💡 Show Price Sensitivity")

    if submitted:
        price_range = np.linspace(base_price * 0.8, base_price * 1.2, num=6)
        results = []
        for price in price_range:
            rel_price = price / risk_amount if risk_amount != 0 else 0
            row = pd.DataFrame([[risk_amount, rel_price]], columns=['Normalized Risk Amount', 'Relative Price'])
            for col in X.columns:
                if col not in row.columns:
                    row[col] = 0
            for col in X.columns:
                if f"Country_{country}" == col or f"Sector_{sector}" == col or f"Broker_{broker}" == col:
                    row[col] = 1
            prob = model.predict_proba(row[X.columns])[:, 1][0]
            results.append({"Quoted Price ($)": price, "Estimated Win Probability (%)": prob * 100})

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.format({"Quoted Price ($)": "${:,.2f}", "Estimated Win Probability (%)": "{:.2f}%"}))

        optimal_row = results_df.loc[results_df['Estimated Win Probability (%)'].idxmax()]

        st.markdown(
            f"<div style='background-color:#d6eaf8;padding:10px;border-radius:5px;'>"
            f"<b>🔵 Suggested Optimal Premium:</b> ${optimal_row['Quoted Price ($)']:.2f}<br>"
            f"<b>🎯 Estimated Win Probability at this price:</b> {optimal_row['Estimated Win Probability (%)']:.2f}%"
            f"</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.lineplot(data=results_df, x='Quoted Price ($)', y='Estimated Win Probability (%)', marker="o", ax=ax)
        ax.axvline(optimal_row['Quoted Price ($)'], color='green', linestyle='--', label='Optimal Price')
        ax.set_title("Quoted Price vs Estimated Win Probability", fontsize=10)
        ax.set_ylabel("Win Probability (%)", fontsize=8)
        ax.set_xlabel("Quoted Price ($)", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.header("🔥 Burn Rate Insights")

    claims_df['BurnRate'] = claims_df['ClaimsReserve'] / claims_df['GWP']
    burn_summary = claims_df.groupby(['Country', 'Sector', 'Broker'])['BurnRate'].mean().reset_index()

    st.sidebar.header("🔍 Filter Burn Rate")
    country_filter = st.sidebar.multiselect("Country", options=burn_summary['Country'].unique(), default=burn_summary['Country'].unique())
    sector_filter = st.sidebar.multiselect("Sector", options=burn_summary['Sector'].unique(), default=burn_summary['Sector'].unique())
    filtered_burn = burn_summary[(burn_summary['Country'].isin(country_filter)) & (burn_summary['Sector'].isin(sector_filter))]

    st.subheader("📄 Burn Rate Summary Table")
    st.dataframe(filtered_burn)

    st.subheader("📉 Burn Rate Heatmap by Sector and Country")
    heatmap_data = filtered_burn.groupby(['Sector', 'Country'])['BurnRate'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Sector', columns='Country', values='BurnRate')
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax2)
    st.pyplot(fig2)

else:
    st.warning("Please upload both Quotes and Claims CSV files to proceed.")
