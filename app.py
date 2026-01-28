import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# -------------------- LOAD CSS --------------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------- TITLE & DESCRIPTION --------------------
st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    """
    **This system uses K-Means Clustering to group customers based on their purchasing behavior and similarities.**

    👉 *Discover hidden customer groups without predefined labels.*
    """
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Wholesale customers data.csv")
    return df

df = load_data()

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox("Select Feature 1", numeric_columns)
feature_2 = st.sidebar.selectbox("Select Feature 2", numeric_columns)

k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

random_state = st.sidebar.number_input(
    "Random State (Optional)", value=42, step=1
)

run = st.sidebar.button("🟦 Run Clustering")

# -------------------- MAIN LOGIC --------------------
if run:
    selected_features = df[[feature_1, feature_2]]

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_features)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_data)

    df_result = df.copy()
    df_result["Cluster"] = clusters

    # -------------------- VISUALIZATION --------------------
    st.subheader("📊 Customer Clusters Visualization")

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        selected_features[feature_1],
        selected_features[feature_2],
        c=clusters,
        cmap="viridis"
    )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=200,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.legend()

    st.pyplot(fig)

    # -------------------- CLUSTER SUMMARY --------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df_result
        .groupby("Cluster")[[feature_1, feature_2]]
        .agg(["count", "mean"])
    )

    summary.columns = [
        "Data Points",
        f"Avg {feature_1}",
        "Data Points (dup)",
        f"Avg {feature_2}"
    ]
    summary = summary.drop(columns=["Data Points (dup)"])

    st.dataframe(summary, use_container_width=True)

    # -------------------- BUSINESS INTERPRETATION --------------------
    st.subheader("💡 Business Interpretation")

    for cluster_id in range(k):
        avg_f1 = df_result[df_result["Cluster"] == cluster_id][feature_1].mean()
        avg_f2 = df_result[df_result["Cluster"] == cluster_id][feature_2].mean()

        st.markdown(
            f"""
            🟢 **Cluster {cluster_id}**  
            Customers in this group show similar behavior with  
            **average {feature_1}: {avg_f1:.2f}** and  
            **average {feature_2}: {avg_f2:.2f}**.
            """
        )

    # -------------------- USER GUIDANCE --------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour and "
        "can be targeted with similar business strategies."
    )

else:
    st.warning("👈 Please select features and click **Run Clustering** to view results.")
