
# insurance_pricing_tool.py (Streamlit Cloud-Ready with Filtering and Export)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(layout="wide")

st.title("📊 Insurance Pricing Intelligence Dashboard")

# Upload CSV files
quotes_file = st.file_uploader("Upload Quotes CSV", type="csv")
claims_file = st.file_uploader("Upload Claims CSV", type="csv")

if quotes_file and claims_file:
    quotes_df = pd.read_csv(quotes_file)
    claims_df = pd.read_csv(claims_file)

    # Encode Result
    quotes_df['Result'] = quotes_df['Result'].map({'Win': 1, 'Loss': 0})

    # Encode categorical variables
    quotes_encoded = pd.get_dummies(quotes_df, columns=['Country', 'Sector', 'Broker'])

    # Train Win/Loss Model
    X = quotes_encoded.drop(['QuoteID', 'Result'], axis=1)
    y = quotes_encoded['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Win/Loss Prediction Model Accuracy: {accuracy:.2%}")

    # Burn Rate Calculation
    claims_df['BurnRate'] = claims_df['ClaimsReserve'] / claims_df['GWP']
    burn_summary = claims_df.groupby(['Country', 'Sector', 'Broker'])['BurnRate'].mean().reset_index()

    # Chart Filters
    st.sidebar.header("🔍 Filter Burn Rate")
    country_filter = st.sidebar.multiselect("Country", options=burn_summary['Country'].unique(), default=burn_summary['Country'].unique())
    sector_filter = st.sidebar.multiselect("Sector", options=burn_summary['Sector'].unique(), default=burn_summary['Sector'].unique())

    filtered_burn = burn_summary[(burn_summary['Country'].isin(country_filter)) & (burn_summary['Sector'].isin(sector_filter))]

    st.subheader("🔥 Burn Rate Summary")
    st.dataframe(filtered_burn)

    # Export to Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        filtered_burn.to_excel(writer, index=False, sheet_name='BurnRate')
    st.download_button("📥 Download Burn Rate (Excel)", data=excel_buffer.getvalue(), file_name="burn_rate.xlsx")

    # Burn Rate Heatmap
    st.subheader("📉 Burn Rate Heatmap by Sector and Country")
    heatmap_data = filtered_burn.pivot(index='Sector', columns='Country', values='BurnRate')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Reds", ax=ax)
    st.pyplot(fig)

    # Export heatmap as PDF
    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format='pdf')
    st.download_button("📥 Download Heatmap (PDF)", data=pdf_buffer.getvalue(), file_name="burn_rate_heatmap.pdf")

    # Win Rate by Price Bracket
    st.subheader("💡 Win Rate by Price Bracket")
    quotes_df['PriceBracket'] = pd.cut(quotes_df['PriceQuoted'], bins=5)
    win_rate_by_price = quotes_df.groupby('PriceBracket')['Result'].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(data=win_rate_by_price, x='PriceBracket', y='Result', palette="Blues_d", ax=ax2)
    ax2.set_ylabel("Win Rate")
    ax2.set_xlabel("Price Bracket")
    ax2.set_title("Win Rate by Price Bracket")
    st.pyplot(fig2)

    # AI Recommendation Interface
    st.subheader("🤖 AI Pricing Recommendation")
    with st.form("recommendation_form"):
        country = st.selectbox("Country", sorted(quotes_df['Country'].unique()))
        sector = st.selectbox("Sector", sorted(quotes_df['Sector'].unique()))
        broker = st.selectbox("Broker", sorted(quotes_df['Broker'].unique()))
        price = st.number_input("Quoted Price", min_value=0.0, value=5000.0)
        submitted = st.form_submit_button("Get Recommendation")

    if submitted:
        row = pd.DataFrame([[price]], columns=['PriceQuoted'])
        for col in X.columns:
            if col not in row.columns:
                row[col] = 0
        for col in X.columns:
            if f"Country_{country}" == col or f"Sector_{sector}" == col or f"Broker_{broker}" == col:
                row[col] = 1
        prob = model.predict_proba(row[X.columns])[:, 1][0]
        st.info(f"Estimated win probability at price {price}: {prob:.2%}")

else:
    st.warning("Please upload both Quotes and Claims CSV files to proceed.")
