import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# LOAD FILE
# ================================
df_trans = pd.read_csv("01_data_full_cluster.csv")
df_rfm = pd.read_csv("02_rfm_cluster.csv")
df_summary = pd.read_csv("03_cluster_summary.csv")
df_pred = pd.read_csv("prediction_results_walkforward.csv")

# Pastikan kolom Month, Year ada
if "Month" in df_pred.columns and "Year" in df_pred.columns:
    df_pred["Period"] = df_pred["Year"].astype(str) + "-" + df_pred["Month"].astype(str).str.zfill(2)
else:
    df_pred["Period"] = ""

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
        size=df_rfm["Monetary"].abs(),     # aman dari nilai negatif
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

    st.subheader("Actual vs Predicted Revenue")
    fig7 = px.scatter(
        df_pred,
        x="Actual_Revenue",
        y="Predicted_Revenue",
        title="Actual vs Predicted"
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Revenue Trend (Actual vs Predicted per Month)")
    fig8 = px.line(
        df_pred,
        x="Period",
        y=["Actual_Revenue", "Predicted_Revenue"],
        markers=True,
        title="Monthly Revenue Trend"
    )
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Prediction Data")
    st.dataframe(df_pred)

    # Error metrics
    st.subheader("Error Metrics")
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(df_pred["Actual_Revenue"], df_pred["Predicted_Revenue"])
    mse = mean_squared_error(df_pred["Actual_Revenue"], df_pred["Predicted_Revenue"])

    st.write(f"**MAE:** {mae:,.2f}")
    st.write(f"**MSE:** {mse:,.2f}")

# ==========================================
# BAGIAN ASNA: REKOMENDASI STRATEGI BISNIS
# ==========================================
st.markdown("---")
st.header("🤖 AI-Driven Business Recommendations")
st.caption("Rekomendasi strategi otomatis berdasarkan hasil analisis data historis.")

# Kita bagi strategi jadi 3 Tab biar rapi
tab1, tab2, tab3 = st.tabs(["📦 Inventory & Stok", "📢 Marketing & Promo", "🛒 Bundling Produk"])

with tab1:
    st.subheader("Strategi Manajemen Stok (Seasonal)")
    col_inv1, col_inv2 = st.columns(2)
    
    with col_inv1:
        st.warning("⚠️ **Alert: Persiapan Peak Season**")
        st.write("""
        * **Insight:** Data menunjukkan tren penjualan selalu melonjak drastis di bulan **November & Desember**.
        * **Action:** Tingkatkan stok produk *Best Seller* (seperti **'White Hanging Heart T-Light Holder'**) sebesar **30%** mulai bulan Oktober untuk mencegah kehabisan stok.
        """)
        
    with col_inv2:
        st.info("📉 **Alert: Antisipasi Low Season**")
        st.write("""
        * **Insight:** Prediksi penjualan turun signifikan di bulan **Januari**.
        * **Action:** Hindari restock besar-besaran di akhir Desember. Siapkan event **'Cuci Gudang Awal Tahun'** untuk produk sisa Natal.
        """)

with tab2:
    st.subheader("Strategi Target Segmen (Clustering)")
    col_mkt1, col_mkt2 = st.columns(2)
    
    with col_mkt1:
        st.success("💎 **Target: Cluster 2 (VIP Clients)**")
        st.write("""
        * **Karakter:** Jarang belanja, tapi nominal transaksi besar.
        * **Action:** Jangan kirim spam diskon receh. Tawarkan **Layanan Prioritas** atau **Akses Pre-Order Eksklusif** lewat WhatsApp pribadi.
        """)
        
    with col_mkt2:
        st.error("💸 **Target: Cluster 3 (Thrifty Shoppers)**")
        st.write("""
        * **Karakter:** Sering belanja (43x) tapi nilai keranjang kecil (£1.3k).
        * **Action:** Terapkan **Minimum Pembelian £20 untuk Gratis Ongkir**. Ini memaksa mereka menambah 1 barang lagi tiap checkout.
        """)

with tab3:
    st.subheader("Strategi Produk (Cross-Selling)")
    st.info("💡 **Peluang Bundling Produk**")
    st.markdown("""
    Berdasarkan pola pembelian, pelanggan sering membeli barang ini bersamaan:
    
    1. **Paket Baking:** *Teatime Fairy Cake Cases* + *Pack of 72 Retrospot Cake Cases*
       * 👉 **Strategi:** Jual sebagai satu paket "Home Baking Kit" dengan diskon 10%.
       
    2. **Paket Dekorasi:** *Wooden Heart Decoration* + *Wooden Star Decoration*
       * 👉 **Strategi:** Tawarkan *Wooden Star* di halaman checkout saat user membeli *Wooden Heart*.
    """)