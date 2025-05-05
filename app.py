import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #0099ff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown("<h1 style='text-align: center; color: #003566;'>🧠 Customer Segmentation using KMeans</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---- Sidebar ----
with st.sidebar:
    st.header("📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload customer data", type=["csv"])
    st.markdown("---")
    st.header("⚙️ Clustering Config")

    num_clusters = st.slider("Select number of clusters", 2, 10, 3)
    show_raw = st.checkbox("Show Raw Data", value=True)

# ---- Main Logic ----
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean data
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df.fillna(numeric_df.mean(), inplace=True)

    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # KMeans
    model = KMeans(n_clusters=num_clusters, random_state=42)
    labels = model.fit_predict(scaled_data)
    df['Cluster'] = labels

    # --- Display Data ---
    if show_raw:
        st.subheader("📊 Raw Data")
        st.dataframe(df, use_container_width=True)

    # --- Cluster Plot ---
    st.subheader("📈 Cluster Visualization")

    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=scaled_data[:, 0], y=scaled_data[:, 1],
            hue=labels, palette='tab10', s=100, ax=ax
        )
        ax.set_xlabel(numeric_df.columns[0])
        ax.set_ylabel(numeric_df.columns[1])
        ax.set_title("Customer Segmentation (2D Projection)")
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numerical features to show cluster visualization.")

    # --- Cluster Size ---
    st.subheader("📦 Cluster Summary")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    st.dataframe(cluster_counts, use_container_width=True)

    # --- Summary Stats ---
    st.subheader("📋 Cluster-wise Statistics")
    st.dataframe(df.groupby("Cluster").mean(numeric_only=True), use_container_width=True)

else:
    st.warning("👈 Please upload a CSV file to start segmenting customers.")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Designed By Karthik using Streamlit | KMeans Clustering</p>",
    unsafe_allow_html=True
)
