import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from utils.preprocessing import initial_cleaning, process_nlp, engineer_features


def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""


bg_image_base64 = get_base64_of_bin_file("prod/assets/processed_bg.jpg")

st.set_page_config(page_title="Airline Review Dashboard", page_icon="✈️", layout="wide")

# Enhanced Premium CSS Design
st.markdown(
    """
<link rel="stylesheet" href="styles/streamlit_theme.css">
""",
    unsafe_allow_html=True,
)

if bg_image_base64:
    # Apply the chosen image as a full-bleed background across the entire viewport
    # (html/body and Streamlit app container). We keep a subtle white overlay so text remains readable.
    st.markdown(
        f"""
        <style>
        /* Full-bleed background for the entire app viewport */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div {{
            background-image: linear-gradient(rgba(255,255,255,0.28), rgba(255,255,255,0.28)), url("data:image/jpeg;base64,{bg_image_base64}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            min-height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
            background-repeat: no-repeat !important;
        }}

        /* Make the central block container transparent so the page background shows edge-to-edge */
        .block-container {{
            background: transparent !important;
            padding: 40px !important;
        }}

        /* Keep the sidebar visually separate and on top */
        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg) !important;
            z-index: 2 !important;
        }}

        /* Ensure hero section is transparent and uses theme text color */
        .hero-section {{
            background: transparent !important;
            padding: 2.5rem 1.5rem !important;
        }}
        .hero-section h1, .hero-section .hero-sub {{
            color: var(--text-color) !important;
            text-shadow: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero-section">
  <h1>✈️ Airline Review Data Engine</h1>
  <p class="hero-sub">An intelligent, end-to-end preprocessing pipeline for aviation customer feedback.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Session state initialization
states = ["raw_data", "cleaned_data", "nlp_data", "final_data"]
for state in states:
    if state not in st.session_state:
        st.session_state[state] = None

# Sidebar Navigation
with st.sidebar:
    # local logo from Pencil export
    st.image("prod/assets/components/prUUl.png", width=80)
    st.header("Pipeline Navigator")
    st.markdown("---")

    stages = [
        "1. Data Ingestion",
        "2. Structural Cleaning",
        "3. NLP Processing",
        "4. Feature Engineering",
        "5. Final Export",
    ]

    choice = st.radio("Select Stage:", stages, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Powered by Streamlit & Pandas")


# Helper function to render metric cards
def render_metric(title, value):
    return f"""
    <div class='metric-card'>
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>
    """


# Main Content Routing
if choice == "1. Data Ingestion":
    st.header("Upload Airline Reviews Dataset")
    st.markdown(
        "Ingest raw CSV data (e.g., Skytrax reviews) to begin the preprocessing pipeline."
    )

    uploaded_file = st.file_uploader(
        "Drop your dataset here or click to browse", type="csv"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing dataset structure..."):
                df = pd.read_csv(uploaded_file)
                st.session_state["raw_data"] = df

            st.success("✅ Dataset successfully ingested!")

            # Key Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    render_metric("Total Flights/Reviews", f"{df.shape[0]:,}"),
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    render_metric("Features (Columns)", df.shape[1]),
                    unsafe_allow_html=True,
                )
            with col3:
                missing_pct = round((df.isnull().sum().sum() / df.size) * 100, 2)
                st.markdown(
                    render_metric("Data Sparsity", f"{missing_pct}%"),
                    unsafe_allow_html=True,
                )

            # Preview Tabs
            tab1, tab2 = st.tabs(["Raw Data Preview", "Data Types Explorer"])
            with tab1:
                st.dataframe(df.head(15), use_container_width=True)
            with tab2:
                types_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index()
                types_df.columns = ["Feature Name", "Data Type"]
                st.dataframe(types_df, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to ingest data: {e}")

elif choice == "2. Structural Cleaning":
    st.header("Structural Data Cleansing")
    st.markdown(
        "Handle missing values, extract verification status, and drop sparse columns automatically."
    )

    if st.session_state["raw_data"] is not None:
        if st.button("🚀 Execute Cleaning Pipeline"):
            with st.spinner("Applying cleaning algorithms..."):
                df_clean = initial_cleaning(st.session_state["raw_data"])
                st.session_state["cleaned_data"] = df_clean

            st.success("✨ Dataset optimized and cleaned!")

            st.subheader("Transformation Report")
            st.markdown("### 📊 Missing Values: Before vs After")

            # Prepare data for matplotlib
            before_missing = st.session_state["raw_data"].isnull().sum()
            after_missing = df_clean.isnull().sum()

            # Align matching index
            features = list(before_missing.index)
            missing_b = before_missing.values
            # After cleaning, some columns might be dropped, so let's match indexes
            missing_a = [
                after_missing[col] if col in after_missing.index else 0
                for col in features
            ]

            # Filter features to only those that had missing values originally to avoid cluttering
            features_to_plot = [f for i, f in enumerate(features) if missing_b[i] > 0]
            if not features_to_plot:
                features_to_plot = features[:10]  # fall back if none missing

            filtered_missing_b = [
                missing_b[features.index(f)] for f in features_to_plot
            ]
            filtered_missing_a = [
                missing_a[features.index(f)] for f in features_to_plot
            ]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            x = np.arange(len(features_to_plot))
            width = 0.35

            ax.bar(
                x - width / 2,
                filtered_missing_b,
                width,
                label="Before Cleaning",
                color="#ff6e40",
            )
            ax.bar(
                x + width / 2,
                filtered_missing_a,
                width,
                label="After Cleaning",
                color="#4A90E2",
            )

            ax.set_ylabel("Number of Missing Values")
            ax.set_title(
                "Missing Values Profile (Features with Missing Data)",
                fontsize=14,
                pad=15,
                color="#333",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(features_to_plot, rotation=45, ha="right")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            # Add value labels
            for p1, p2 in zip(
                ax.patches[: len(features_to_plot)], ax.patches[len(features_to_plot) :]
            ):
                ax.annotate(
                    str(p1.get_height()),
                    (p1.get_x() + p1.get_width() / 2.0, p1.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333",
                )
                ax.annotate(
                    str(p2.get_height()),
                    (p2.get_x() + p2.get_width() / 2.0, p2.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333",
                )

            # Reduce border visibility
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig)

            st.markdown("### Cleaned Data Snapshot")
            st.dataframe(df_clean.head(10), use_container_width=True)

            # Actionable insight
            dropped_cols = set(st.session_state["raw_data"].columns) - set(
                df_clean.columns
            )
            if dropped_cols:
                st.info(f"**Dropped high-sparsity columns:** {', '.join(dropped_cols)}")
    else:
        st.info("👈 Please ingest data in Stage 1 to proceed.")

elif choice == "3. NLP Processing":
    st.header("Natural Language Processing (NLP)")
    st.markdown(
        "Isolate verbatim feedback, standardize casing, and execute foundational text cleaning."
    )

    if st.session_state["cleaned_data"] is not None:
        df_target = st.session_state["cleaned_data"]
        text_cols = [
            col for col in df_target.columns if df_target[col].dtype == "object"
        ]

        st.markdown("### Configuration")
        text_column = st.selectbox(
            "Select Target Verbatim Feature",
            options=text_cols,
            help="Select the column containing the primary passenger review text.",
        )

        if st.button("🧠 Execute NLP Pipeline"):
            if text_column:
                with st.spinner("Tokenizing and standardizing linguistic data..."):
                    df_nlp = process_nlp(df_target, text_column)
                    st.session_state["nlp_data"] = df_nlp

                st.success("✨ Linguistic data successfully processed!")

                st.markdown("### NLP Transformation Trace")
                st.dataframe(
                    df_nlp[
                        [
                            text_column,
                            f"{text_column}_cleaned",
                            f"{text_column}_word_count",
                        ]
                    ].head(15),
                    use_container_width=True,
                )
            else:
                st.warning("Please select a valid text column.")
    else:
        st.info("👈 Please complete Stage 2 (Structural Cleaning) to proceed.")

elif choice == "4. Feature Engineering":
    st.header("Feature Engineering & Encoding")
    st.markdown(
        "Transform categorical variables into machine-readable numerical representations."
    )

    # Prioritize NLP data, fallback to cleaned data
    source_data = (
        st.session_state["nlp_data"]
        if st.session_state["nlp_data"] is not None
        else st.session_state["cleaned_data"]
    )

    if source_data is not None:
        st.info(
            "Automatically detecting low-cardinality categorical variables for One-Hot Encoding."
        )

        if st.button("⚙️ Execute Feature Engineering"):
            with st.spinner("Applying encoding manifolds..."):
                df_engineered = engineer_features(source_data)
                st.session_state["final_data"] = df_engineered

            st.success("✨ Feature space successfully engineered for ML models!")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    render_metric("Original Features", source_data.shape[1]),
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    render_metric("Engineered Features", df_engineered.shape[1]),
                    unsafe_allow_html=True,
                )

            st.markdown("### Engineered Matrix Preview")
            st.dataframe(df_engineered.head(10), use_container_width=True)
    else:
        st.info("👈 Please complete previous stages to proceed.")

elif choice == "5. Final Export":
    st.header("Final Integration and Export")
    st.markdown(
        "Review the end-to-end processed dataset and serialize it for downstream machine learning applications."
    )

    if st.session_state["final_data"] is not None:
        final_df = st.session_state["final_data"]

        st.markdown("### Golden Record Dataset")
        st.dataframe(final_df, use_container_width=True)

        st.markdown("### Export Hub")

        col1, col2 = st.columns(2)
        with col1:
            csv_buffer = io.StringIO()
            final_df.to_csv(csv_buffer, index=False)
            btn = st.download_button(
                label="📥 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name="airline_reviews_ml_ready.csv",
                mime="text/csv",
            )
        with col2:
            st.info(
                "The exported dataset is now fully encoded, scrubbed of nulls, and prepared for direct ingestion into modern ML classifiers (e.g., Scikit-Learn Random Forests, XGBoost)."
            )
    else:
        st.info(
            "👈 The pipeline is incomplete. Please finish stages 1-4 to generate the final export."
        )
