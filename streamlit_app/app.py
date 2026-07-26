import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🎬 MovieIQ",
    page_icon="🎥",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "output", "clean_movies.csv")
    return pd.read_csv(data_path)

df = load_data()
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Features
# -----------------------------
X = df[["budget", "popularity", "runtime", "vote_average"]]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📋 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 EDA Dashboard",
        "📈 Statistical Tests",
        "🎯 Prediction",
        "⭐ Feature Importance"
    ]
)

# ===================================================
# HOME PAGE
# ===================================================
if page == "🏠 Home":

    st.title("🎬 MovieIQ")
    st.subheader("Predictive Analytics on Film Success")

    st.markdown("---")

    st.write("""
MovieIQ predicts whether a movie is likely to be successful based on:

- Budget
- Popularity
- Runtime
- Vote Average

A movie is considered successful when:

Revenue > Budget
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Movies", len(df))
    col2.metric("Successful Movies", int(df["success"].sum()))
    col3.metric("Average Rating", round(df["vote_average"].mean(), 2))

    st.success("Dataset Loaded Successfully ✅")
# ===================================================
# EDA DASHBOARD
# ===================================================
elif page == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    # Sidebar Filters
    genre_list = sorted(df["genres"].dropna().unique())

    selected_genre = st.sidebar.selectbox(
        "Select Genre",
        ["All"] + genre_list
    )

    min_vote = st.sidebar.slider(
        "Minimum Vote Average",
        float(df["vote_average"].min()),
        float(df["vote_average"].max()),
        float(df["vote_average"].min())
    )

    filtered_df = df.copy()

    if selected_genre != "All":
        filtered_df = filtered_df[
            filtered_df["genres"].str.contains(
                selected_genre,
                case=False,
                na=False
            )
        ]

    filtered_df = filtered_df[
        filtered_df["vote_average"] >= min_vote
    ]

    st.write(f"Filtered Movies: {len(filtered_df)}")

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Dataset",
        data=csv,
        file_name="filtered_movies.csv",
        mime="text/csv"
    )

    st.markdown("---")

    st.subheader("Budget vs Revenue")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        filtered_df["budget"],
        filtered_df["revenue"],
        alpha=0.5
    )
    ax.set_xlabel("Budget")
    ax.set_ylabel("Revenue")
    ax.set_title("Budget vs Revenue")
    st.pyplot(fig)

    st.subheader("Movie Success Distribution")

    fig, ax = plt.subplots()
    sns.countplot(
        data=filtered_df,
        x="success",
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Popularity Distribution")

    fig, ax = plt.subplots()
    sns.histplot(
        filtered_df["popularity"],
        kde=True,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Runtime Distribution")

    fig, ax = plt.subplots()
    sns.histplot(
        filtered_df["runtime"],
        kde=True,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Vote Average Distribution")

    fig, ax = plt.subplots()
    sns.histplot(
        filtered_df["vote_average"],
        kde=True,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = filtered_df.select_dtypes(include="number").corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
# ===================================================
# STATISTICAL TESTS
# ===================================================
elif page == "📈 Statistical Tests":

    st.title("📈 Statistical Test Results")

    st.subheader("T-Test Results")

    t_stat = 2.0647
    p_value = 0.0391

    st.write(f"**T-Statistic:** {t_stat}")
    st.write(f"**P-Value:** {p_value}")

    if p_value < 0.05:
        st.success(
            "Popularity is significantly different between successful and unsuccessful movies."
        )
    else:
        st.error("No significant difference found.")

    st.markdown("---")

    st.subheader("Chi-Square Test")

    st.info("""
Genre and Movie Success are significantly associated.

P-Value < 0.05

Hence, Genre has a statistically significant association with Movie Success.
""")

    st.markdown("---")

    st.subheader("Business Interpretation")

    st.write("""
- Popular movies are more likely to become successful.
- Genre also has a significant impact on movie success.
- Budget and popularity are useful predictors for the machine learning model.
""")
# ===================================================
# PREDICTION PAGE
# ===================================================
elif page == "🎯 Prediction":

    st.title("🎯 Movie Success Prediction")

    budget = st.number_input(
        "Budget",
        min_value=0.0,
        value=1000000.0
    )

    popularity = st.number_input(
        "Popularity",
        min_value=0.0,
        value=50.0
    )

    runtime = st.number_input(
        "Runtime (Minutes)",
        min_value=1,
        value=120
    )

    vote_average = st.slider(
        "Vote Average",
        min_value=0.0,
        max_value=10.0,
        value=6.5
    )

    if st.button("Predict Movie Success"):

        input_data = pd.DataFrame({
            "budget": [budget],
            "popularity": [popularity],
            "runtime": [runtime],
            "vote_average": [vote_average]
        })

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("🎉 This movie is likely to be Successful!")
        else:
            st.error("❌ This movie is likely to be Unsuccessful.")

        st.subheader("Input Summary")

        st.write(input_data)
# ===================================================
# FEATURE IMPORTANCE
# ===================================================
elif page == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("Feature Importance Table")
    st.dataframe(importance, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")

    plt.xticks(rotation=20)

    st.pyplot(fig)

    csv = importance.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Feature Importance",
        data=csv,
        file_name="feature_importance.csv",
        mime="text/csv"
    )



