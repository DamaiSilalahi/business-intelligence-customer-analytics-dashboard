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
# PAGE 4 — BUSINESS STRATEGY (KOMBINASI FINAL: VISUAL GAMBAR + STRATEGI TABS)
# ==============================================================================
elif page == "Business Strategy":
    st.title("💡 Interpretasi Data & Strategi Bisnis")
    st.markdown("""
    Halaman ini merangkum **penjelasan output (insight)** dari setiap halaman dashboard serta **rekomendasi strategi bisnis** yang dapat dieksekusi.
    """)
    st.markdown("---")

    # =======================================================
    # BAGIAN 1: CLUSTERING (Sesuai Gambar Blue Box)
    # =======================================================
    st.header("1️⃣ Penjelasan Output Clustering (Halaman 1)")
    
    # 1. Penjelasan Output (Persis Text Gambar Biru)
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("📊 Grafik: Distribusi & Monetary")
        st.info("""
        **1. Pie Chart (Distribusi Pelanggan):**
        * **Output:** Terlihat mayoritas pelanggan masuk ke **Cluster 3 (Hemat)**, sedangkan **Cluster 2 (VIP)** jumlahnya paling sedikit.
        * **Makna:** Bisnis ini ditopang oleh banyak pembeli kecil, bukan sedikit pembeli besar (Volume Based).
        
        **2. Bar Chart (Rata-rata Monetary):**
        * **Output:** Batang Cluster 2 menjulang paling tinggi.
        * **Makna:** Meskipun jumlah orangnya sedikit, Cluster 2 menyumbang rata-rata uang terbesar per orang.
        """)
        
    with col_c2:
        st.subheader("📈 Grafik: RFM Scatter Plot")
        st.info("""
        **3. Scatter Plot (Sebaran Titik):**
        * **Output:** Titik-titik pelanggan terkelompok jelas berdasarkan warna.
        * **Makna:** Algoritma K-Means berhasil memisahkan pelanggan secara valid.
            * Area **Kanan Atas:** Pelanggan sering belanja & baru saja belanja (Loyal).
            * Area **Kiri Bawah:** Pelanggan jarang belanja & sudah lama hilang (Churn).
        """)

    # 2. Strategi Bisnis (Tabs - VIP Red Carpet)
    st.write("👉 **Rekomendasi Strategi (Klik Tab di bawah):**")
    
    tab_vip, tab_loyal, tab_hemat, tab_churn = st.tabs(["👑 VIP Clients", "🌟 Loyal Customers", "🛒 Hemat/Thrifty", "⚠️ Berisiko Churn"])

    with tab_vip:
        st.success("""
        **Cluster 2: VIP / Big Spenders**
        * **Karakter:** Jarang belanja, tapi sekali transaksi nilainya sangat besar.
        * **Strategi:** 'The Red Carpet Treatment'
        * **Action:** Tawarkan layanan **Personal Shopper** via WhatsApp.
        * **Offer:** Akses **Pre-Order** eksklusif untuk produk baru (No Discount needed).
        """)

    with tab_loyal:
        st.info("""
        **Cluster 0: Loyal Customers**
        * **Karakter:** Rutin belanja dengan nilai transaksi menengah.
        * **Strategi:** 'Lock-in Ecosystem'
        * **Action:** Implementasikan **Point Reward System**. Setiap 10x belanja gratis 1 produk sample.
        """)

    with tab_hemat:
        st.warning("""
        **Cluster 3: Thrifty (Si Hemat)**
        * **Karakter:** Sering belanja, tapi nilai keranjangnya kecil (receh).
        * **Strategi:** 'Increase Basket Size'
        * **Action:** Terapkan aturan **"Gratis Ongkir Min. Belanja £20"** untuk menaikkan margin.
        """)

    with tab_churn:
        st.error("""
        **Cluster 1: Churn Risk**
        * **Karakter:** Sudah sangat lama tidak kembali belanja.
        * **Strategi:** 'Win-Back Campaign'
        * **Action:** Kirim email otomatis berisi **Voucher Diskon 20%** yang hangus dalam 24 jam.
        """)

    st.markdown("---")

    # =======================================================
    # BAGIAN 2: TRANSAKSI & PRODUK (Sesuai Gambar Yellow Box)
    # =======================================================
    st.header("2️⃣ Penjelasan Output Transaksi (Halaman 2)")
    
    # 1. Penjelasan Output (Persis Text Gambar Kuning)
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("🌍 Grafik: Sales per Country")
        st.warning("""
        **Output:** Batang 'United Kingdom' mendominasi total penjualan (>90%).
        
        **Insight:** Pasar internasional belum maksimal. Strategi marketing saat ini sebaiknya fokus defensif di pasar lokal (UK) karena merupakan sumber pendapatan utama.
        """)
        
    with col_t2:
        st.subheader("📦 Grafik: Top 10 Products")
        st.warning("""
        **Output:** Produk *'White Hanging Heart T-Light Holder'* adalah produk terlaris #1.
        
        **Insight:** Ini adalah 'Produk Pancingan'. Pastikan stok barang ini **tidak boleh kosong** karena sering menjadi pintu masuk pelanggan untuk membeli barang lain.
        """)

    # 2. Strategi Bundling (Sesuai Gambar Paket Baking)
    st.write("👉 **Rekomendasi Bundling Produk (Cross-Selling):**")
    
    col_bund1, col_bund2 = st.columns(2)
    
    with col_bund1:
        st.success("""
        **1. Paket Baking (Home Baking Kit)**
        * **Isi:** *Teatime Fairy Cake Cases* + *Pack of 72 Retrospot Cake Cases*
        * **Strategi:** Jual sebagai satu paket dengan diskon 10%.
        """)
        
    with col_bund2:
        st.success("""
        **2. Paket Dekorasi**
        * **Isi:** *Wooden Heart Decoration* + *Wooden Star Decoration*
        * **Strategi:** Tawarkan *Wooden Star* di halaman checkout saat user membeli *Wooden Heart*.
        """)

    st.markdown("---")

    # =======================================================
    # BAGIAN 3: PREDIKSI REGRESI (Sesuai Gambar Red Box)
    # =======================================================
    st.header("3️⃣ Penjelasan Output Prediksi (Halaman 3)")
    
    # 1. Penjelasan Output (Persis Text Gambar Merah)
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.subheader("📉 Grafik: Actual vs Prediction")
        st.error("""
        **1. Scatter Plot (Akurasi):**
        * **Output:** Titik prediksi membentuk garis lurus diagonal (linear).
        * **Makna:** Model regresi cukup akurat memprediksi angka penjualan (Error Rate rendah).
        """)
        
    with col_r2:
        st.subheader("📅 Grafik: Trend Bulanan")
        st.error("""
        **2. Line Chart (Tren Waktu):**
        * **Output:** Grafik menunjukkan kenaikan tajam di bulan **November-Desember**.
        * **Makna (Seasonality):** Bisnis ini sangat dipengaruhi tren akhir tahun (Natal/Tahun Baru).
        """)

    # 2. Strategi Bisnis (Tabs - Inventory 30%)
    st.write("👉 **Rekomendasi Strategi (Klik Tab di bawah):**")
    
    tab_stok, tab_ops = st.tabs(["📦 Inventory (Stok)", "⚙️ Operasional"])
    
    with tab_stok:
        st.success("""
        **Strategi Manajemen Stok (Seasonal):**
        1. **Q4 (Okt-Des):** Tingkatkan stok barang *Best Seller* sebesar **30%** untuk mencegah *stock-out*.
        2. **Q1 (Jan-Feb):** Tahan pembelian stok baru. Lakukan **Clearance Sale** untuk menghabiskan sisa stok Natal.
        """)
        
    with tab_ops:
        st.info("""
        **Strategi Operasional:**
        1. **Manpower:** Tambah tenaga kerja paruh waktu (*packing*) hanya di bulan November-Desember.
        2. **Efisiensi:** Fokus efisiensi budget operasional di bulan Januari karena *cashflow* masuk menurun.
        """)