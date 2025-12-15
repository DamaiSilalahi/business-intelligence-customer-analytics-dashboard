import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ================================
# CONFIG & LOAD FILE
# ================================
st.set_page_config(page_title="E-Commerce Analytics Dashboard", layout="wide")

# Cache data agar loading cepat
@st.cache_data
def load_data():
    # Pastikan nama file sesuai dengan yang ada di folder kalian
    df_trans = pd.read_csv("01_data_full_cluster.csv")
    df_rfm = pd.read_csv("02_rfm_cluster.csv")
    df_summary = pd.read_csv("03_cluster_summary.csv")
    df_pred = pd.read_csv("prediction_results_walkforward.csv")

    # Feature Engineering sederhana untuk Period
    if "Month" in df_pred.columns and "Year" in df_pred.columns:
        df_pred["Period"] = df_pred["Year"].astype(str) + "-" + df_pred["Month"].astype(str).str.zfill(2)
    else:
        df_pred["Period"] = ""
    
    return df_trans, df_rfm, df_summary, df_pred

# Load Data
try:
    df_trans, df_rfm, df_summary, df_pred = load_data()
except FileNotFoundError:
    st.error("File CSV tidak ditemukan! Pastikan file csv ada di folder yang sama dengan script python.")
    st.stop()

# ================================
# SIDEBAR NAVIGATION
# ================================
st.sidebar.title("📊 Dashboard Navigation")

# ===> PERBAIKAN PENTING DISINI: Menambahkan menu 'Business Strategy' <===
page = st.sidebar.radio("Go to:", [
    "Customer Segmentation (RFM)",
    "Transaction Insights",
    "Sales Prediction (Regression)",
    "Business Strategy"  # <--- INI WAJIB ADA AGAR PAGE 4 MUNCUL
])

st.sidebar.markdown("----")
st.sidebar.caption("Capstone Project Team 5 🔥")

# ==============================================================================
# PAGE 1 — CUSTOMER SEGMENTATION
# ==============================================================================
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
        size=df_rfm["Monetary"].abs(),
        color="Cluster",
        title="RFM Scatter Plot by Cluster",
        opacity=0.7
    )
    st.plotly_chart(fig3, use_container_width=True)

# ==============================================================================
# PAGE 2 — TRANSACTION INSIGHTS
# ==============================================================================
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

    # Top Products
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

# ==============================================================================
# PAGE 3 — SALES PREDICTION (REGRESSION)
# ==============================================================================
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

    st.subheader("Prediction Data & Metrics")
    col_metric1, col_metric2 = st.columns([2, 1])
    
    with col_metric1:
        st.dataframe(df_pred)
    
    with col_metric2:
        mae = mean_absolute_error(df_pred["Actual_Revenue"], df_pred["Predicted_Revenue"])
        mse = mean_squared_error(df_pred["Actual_Revenue"], df_pred["Predicted_Revenue"])
        st.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
        st.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")

# ==============================================================================
# PAGE 4 — BUSINESS STRATEGY (INI BAGIAN ASNA)
# ==============================================================================
elif page == "Business Strategy":
    st.title("💡 Business Insights & Action Plan")
    st.markdown("""
    Halaman ini merangkum **hasil analisis data (Knowledge)** dan menerjemahkannya menjadi 
    **strategi bisnis konkret** untuk pengambilan keputusan.
    """)
    st.markdown("---")

    # -------------------------------------------------------
    # BAGIAN 1: INTERPRETASI HASIL REGRESI (VIRGINIA)
    # -------------------------------------------------------
    st.header("1️⃣ Analisis Tren Penjualan (Regression Insight)")
    
    col_reg1, col_reg2 = st.columns([1, 2])
    
    with col_reg1:
        st.info("📊 **Apa hasil output data ini?**")
        st.markdown("""
        * **Pola Musiman:** Data menunjukkan tren penjualan selalu mencapai puncak (*peak*) di akhir tahun (Nov-Des).
        * **Tren Bulan Depan:** Model regresi memprediksi adanya penurunan permintaan pasca-liburan (Januari).
        """)
        
    with col_reg2:
        st.warning("🚀 **Strategi Bisnis (Action Plan)**")
        st.write("Berdasarkan prediksi tersebut, manajemen disarankan melakukan:")
        
        # Tabs untuk Strategi Inventory
        tab_inv1, tab_inv2 = st.tabs(["📦 Inventory (Stok)", "⚙️ Operasional"])
        
        with tab_inv1:
            st.write("""
            1. **Q4 (Okt-Des):** Tingkatkan stok barang *Best Seller* sebesar **30%** untuk mencegah *stock-out*.
            2. **Q1 (Jan-Feb):** Tahan pembelian stok baru. Lakukan *Clearance Sale* untuk menghabiskan sisa stok Natal.
            """)
        with tab_inv2:
            st.write("""
            1. Tambah tenaga kerja paruh waktu (*packing*) hanya di bulan November-Desember.
            2. Fokus efisiensi budget operasional di bulan Januari karena *cashflow* masuk menurun.
            """)

    st.markdown("---")

    # -------------------------------------------------------
    # BAGIAN 2: INTERPRETASI HASIL CLUSTERING (KATRIN)
    # -------------------------------------------------------
    st.header("2️⃣ Segmentasi Pelanggan (Clustering Insight)")
    st.caption("Analisis RFM membagi pelanggan menjadi 4 karakter unik. Berikut cara menanganinya:")

    tab1, tab2, tab3, tab4 = st.tabs(["👑 VIP Clients", "🌟 Loyal Customers", "🛒 Hemat/Thrifty", "⚠️ Berisiko Churn"])

    with tab1:
        st.subheader("Cluster 2: VIP / Big Spenders")
        st.success("""
        **Karakter:** Jarang belanja, tapi sekali transaksi nilainya sangat besar.
        
        **Strategi: 'The Red Carpet Treatment'**
        * **Action:** Tawarkan layanan **Personal Shopper** via WhatsApp.
        * **Offer:** Akses **Pre-Order** eksklusif untuk produk baru (No Discount needed).
        """)

    with tab2:
        st.subheader("Cluster 0: Loyal Customers")
        st.info("""
        **Karakter:** Rutin belanja dengan nilai transaksi menengah.
        
        **Strategi: 'Lock-in Ecosystem'**
        * **Action:** Implementasikan **Point Reward System**.
        * **Offer:** Setiap 10x belanja gratis 1 produk sample agar mereka tidak pindah kompetitor.
        """)

    with tab3:
        st.subheader("Cluster 3: Thrifty Shoppers (Si Hemat)")
        st.warning("""
        **Karakter:** Sangat sering belanja, tapi nilai keranjangnya kecil (receh).
        
        **Strategi: 'Increase Basket Size'**
        * **Action:** Terapkan aturan **'Gratis Ongkir Min. Belanja £20'**.
        * **Goal:** Memaksa mereka menambah barang ke keranjang demi gratis ongkir (menaikkan margin).
        """)

    with tab4:
        st.subheader("Cluster 1: Churn Risk (Lama Menghilang)")
        st.error("""
        **Karakter:** Sudah sangat lama tidak kembali belanja.
        
        **Strategi: 'Win-Back Campaign'**
        * **Action:** Kirim email otomatis 'We Miss You'.
        * **Offer:** Voucher Diskon 20% yang hangus dalam 24 jam (Urgency).
        """)
    
    st.markdown("---")

    # -------------------------------------------------------
    # BAGIAN 3: CROSS SELLING
    # -------------------------------------------------------
    st.header("3️⃣ Rekomendasi Produk (Bundling Strategy)")
    st.markdown("""
    Berdasarkan pola pembelian, pelanggan sering membeli barang ini bersamaan:
    
    1. **Paket Baking:** *Teatime Fairy Cake Cases* + *Pack of 72 Retrospot Cake Cases*
       * 👉 **Strategi:** Jual sebagai satu paket "Home Baking Kit" dengan diskon 10%.
       
    2. **Paket Dekorasi:** *Wooden Heart Decoration* + *Wooden Star Decoration*
       * 👉 **Strategi:** Tawarkan *Wooden Star* di halaman checkout saat user membeli *Wooden Heart*.
    """)