import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# LOAD FILE
# ================================
df_trans = pd.read_csv("01_data_full_cluster.csv")
df_rfm = pd.read_csv("02_rfm_cluster.csv")
df_summary = pd.read_csv("03_cluster_summary.csv")   
df_pred = pd.read_csv("prediction_results.csv")

# Tambahkan kolom tanggal kosong untuk prediksi
df_pred["Date"] = ""


# ================================
# CONFIG
# ================================
st.set_page_config(page_title="E-Commerce Analytics Dashboard",
                   layout="wide")


# ================================
# SIDEBAR
# ================================
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio("Go to:", [
    "Customer Segmentation (RFM)",
    "Transaction Insights",
    "Sales Prediction (Regression)"
])

st.sidebar.markdown("----")
st.sidebar.caption("Made for your dataset 🔥")


# ================================
# PAGE 1 — CUSTOMER SEGMENTATION
# ================================
if page == "Customer Segmentation (RFM)":
    st.title("🧍 Customer Segmentation — RFM + KMeans")

    st.subheader("Cluster Summary")
    st.dataframe(df_summary)

    col1, col2 = st.columns(2)

    # Pie chart cluster distribution
    with col1:
        fig1 = px.pie(
            df_summary,
            names="Cluster",
            values="Jumlah_Pelanggan",
            title="Distribusi Pelanggan per Cluster"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Bar chart monetary per cluster
    with col2:
        fig2 = px.bar(
            df_summary,
            x="Cluster",
            y="Monetary",
            title="Rata-rata Monetary per Cluster",
            text_auto=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("RFM Scatter Plot")
    fig3 = px.scatter(
        df_rfm,
        x="Recency",
        y="Frequency",
        size=abs(df_rfm["Monetary"]),
        color="Cluster",
        title="RFM Scatter Plot by Cluster",
        opacity=0.7
    )
    st.plotly_chart(fig3, use_container_width=True)


# ================================
# PAGE 2 — TRANSACTION INSIGHTS
# ================================
elif page == "Transaction Insights":
    st.title("🛒 Transaction Insights")

    col1, col2 = st.columns(2)

    # Country sales
    with col1:
        country_sales = df_trans.groupby("Country")["TotalPrice"].sum().reset_index()
        fig4 = px.bar(
            country_sales,
            x="Country",
            y="TotalPrice",
            title="Total Sales per Country"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # # Quantity per cluster
    # with col2:
    #     cluster_qty = df_trans.groupby("Cluster")["Quantity"].sum().reset_index()
    #     fig5 = px.pie(
    #         cluster_qty,
    #         names="Cluster",
    #         values="Quantity",
    #         title="Distribution of Quantity per Cluster"
    #     )
    #     st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Top 10 Products by Revenue")
    top_products = df_trans.groupby("Description")["TotalPrice"] \
                           .sum() \
                           .sort_values(ascending=False) \
                           .head(10) \
                           .reset_index()

    fig6 = px.bar(
        top_products,
        x="TotalPrice",
        y="Description",
        orientation="h",
        title="Top 10 Best-Selling Products"
    )
    st.plotly_chart(fig6, use_container_width=True)


# ================================
# PAGE 3 — SALES PREDICTION (REGRESSION)
# ================================
elif page == "Sales Prediction (Regression)":
    st.title("📈 Sales Prediction — Regression Model")
    st.caption("Kolom tanggal masih dikosongkan → akan diisi setelah model forecasting final.")

    st.subheader("Actual vs Predicted Revenue")
    fig7 = px.scatter(
        df_pred,
        x="Actual_TotalPrice",
        y="Predicted_TotalPrice",
        trendline="ols",
        title="Actual vs Predicted"
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Prediction Data")
    st.dataframe(df_pred)

    # Error metrics (optional)
    st.subheader("Error Metrics")
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(df_pred["Actual_TotalPrice"], df_pred["Predicted_TotalPrice"])
        mse = mean_squared_error(df_pred["Actual_TotalPrice"], df_pred["Predicted_TotalPrice"])

        st.write(f"**MAE:** {mae:,.2f}")
        st.write(f"**MSE:** {mse:,.2f}")

    except:
        st.write("Scikit-learn tidak tersedia — MAE & MSE tidak bisa dihitung otomatis.")


