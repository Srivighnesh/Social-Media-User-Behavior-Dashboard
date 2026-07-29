"""
Social Media User Behavior — Unsupervised Learning Dashboard
==============================================================
A Streamlit dashboard that lets you pick a clustering CATEGORY from the
sidebar and runs that category's full pipeline (feature prep -> elbow
method -> KMeans -> DBSCAN -> PCA visualization -> cluster profiling ->
algorithm comparison), all built on top of a small set of shared,
reusable functions instead of duplicating the pipeline for every category.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Behavior — Unsupervised Dashboard",
    page_icon="ICONS/bulhorn.png",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Category configuration
# Each category defines: the features used for clustering, the KMeans k and
# cluster-name mapping decided from profiling, and the DBSCAN eps/min_samples
# and its own cluster-name mapping. This is the single place you'd edit if
# you retune a category later.
# ----------------------------------------------------------------------------
CATEGORIES = {
    "User Behavior": dict(
        description="How people use the app day-to-day: session length, "
        "frequency, scrolling and notifications.",
        features=[
            "daily_usage_hours",
            "sessions_per_day",
            "avg_session_duration_min",
            "scroll_speed",
            "notification_frequency",
            "takes_social_media_breaks",
            "peak_usage_time",
        ],
        numirical = [
            "daily_usage_hours",
            "sessions_per_day",
            "avg_session_duration_min",
            "scroll_speed",
            "notification_frequency",
        ],

        category = [
            "takes_social_media_breaks",
            "peak_usage_time",
        ],
        kmeans_k=4,
        kmeans_names={
            0: "Night Time Active Users",
            1: "Evening-Afternoon Users",
            2: "Morning Browsers",
            3: "Late-Night Browsers",
        },
        dbscan_eps=3.0,
        dbscan_min_samples=6,
        dbscan_names={
            -1: "Long-Session Explorers",
            0: "Moderate Daily Engagers",
            1: "Core Regular Users",
            2: "Average Session Users",
            3: "Above-Average Daily Users",
        },
        bins={
            "daily_usage_hours": {
                "bins": [0, 1, 3, 6, 24],
                "labels": ["<1 hour", "1-3 hours", "3-6 hours", ">6 hours"]
            },
            "sessions_per_day": {
                "bins": [0, 3, 6, 10, 50],
                "labels": ["<3", "3-6", "6-10", "10-50"]
            },
            "avg_session_duration_min": {
                "bins": [0, 20, 45, 75, 300],
                "labels": ["<20 mins", "20-45 mins", "45-75 mins", ">75 mins"]
            },
        },
    ),
    "Engagement": dict(
        description="How actively people interact with content: posting, "
        "liking, commenting, sharing and mood while scrolling.",
        features=[
            "posts_per_week",
            "likes_given_per_day",
            "comments_per_day",
            "shares_per_day",
            "ad_click_rate",
            "preferred_content_type",
            "mood_while_scrolling",
        ],
        numirical=[
            "posts_per_week",
            "likes_given_per_day",
            "comments_per_day",
            "shares_per_day",
            "ad_click_rate",    
        ],
        category = [
            "preferred_content_type",
            "mood_while_scrolling",
        ],
        kmeans_k=5,
        kmeans_names={
            0: "Educational Engagers",
            1: "Casual Learners",
            2: "Happy Meme Viewers",
            3: "Inspired Social Interactors",
            4: "Entertainment Enthusiasts",
        },
        dbscan_eps=3.0,
        dbscan_min_samples=5,
        dbscan_names={
            -1: "Exploratory Users",
            0: "Visual Content",
            1: "Multi-Content Explorers",
            2: "Video Enthusiasts",
            3: "Entertainment Fans",
            4: "Story Learners",
            5: "News Followers",
        },
        bins={
            'posts_per_week': {
                'bins': [0, 2, 4, 6, 10],
                'labels': ['Low Activity', 'Moderate Activity', 'Active', 'Highly Active']
            },

            'comments_per_day': {
                'bins': [0, 2, 4, 6, 11],
                'labels': ['Low Engagement', 'Moderate Engagement', 'Active', 'Highly Engaged']
            },

            'shares_per_day': {
                'bins': [0, 1, 3, 5, 8],
                'labels': ['Low Sharing', 'Moderate Sharing', 'Active Sharing', 'High Sharing']
            },

            'likes_given_per_day': {
                'bins': [0, 5, 10, 15, 20],
                'labels': ['Low Likes', 'Moderate Likes', 'Active Likes', 'High Likes']
            },
            'ad_click_rate': {  
                
                'bins': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf],
                'labels': ['<1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']
            },
        },
    ),
    "Social Influence": dict(
        description="Reach and influence: followers, following, posting "
        "cadence, influencer status and primary platform.",
        features=[
            "followers_count",
            "following_count",
            "posts_per_week",
            "influencer_status",
            "primary_platform",
        ],
        category=[
            "influencer_status",
            "primary_platform"
        ],
        numirical=[
            "followers_count",
            "following_count",
            "posts_per_week"
        ],
        kmeans_k=5,
        kmeans_names={
            0: "Active Snapchat Users",
            1: "Active YouTube Users",
            2: "Active Instagram Users",
            3: "Active Twitter/X Users",
            4: "Balanced Platform Users",
        },
        dbscan_eps=4.0,
        dbscan_min_samples=6,
        dbscan_names={
            -1: "High-Influence Outliers",
            0: "General Social Media Users",
            1: "LinkedIn Professionals",
            2: "Pinterest Creators",
        },
        bins = {
            'followers_count': {
                'bins': [0, 1000, 5000, 10000, 20000, np.inf],
                'labels': ['<1K', '1K-5K', '5K-10K', '10K-20K', '>20K']
            },

            'following_count': {
                'bins': [0, 200, 500, 1000, 2000, np.inf],
                'labels': ['<200', '200-500', '500-1K', '1K-2K', '>2K']
            },

            'posts_per_week': {
                'bins': [0, 1, 3, 5, 8],
                'labels': ['Low Sharing', 'Moderate Sharing', 'Active Sharing', 'High Sharing']
            },
        },
    ),
    "Spending & Marketing": dict(
        description="Commercial behavior: ad clicks, purchases through "
        "social media and monthly spend.",
        features=[
            "ad_click_rate",
            "purchased_via_social_media",
            "monthly_spend_via_social_usd",
            "primary_purpose",
        ],
        category=[
            "purchased_via_social_media",
            "primary_purpose",
        ],
        numirical=[
            "ad_click_rate",
            "monthly_spend_via_social_usd",
        ],
        kmeans_k=5,
        kmeans_names={
            0: "Low Spenders",
            1: "Occasional Buyers",
            2: "High Spenders",
            3: "Value Buyers",
            4: "Budget Buyers",
        },
        dbscan_eps=2.5,
        dbscan_min_samples=5,
        dbscan_names={
            -1: "Selective High Spenders",
            0: "Entertainment Users",
            1: "News Followers",
            2: "Learners",
            3: "Business Users",
            4: "Social Connectors",
            5: "Network Builders",
        },
        bins={
            'ad_click_rate': {
                'bins': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf],
                'labels': ['<1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']
            },

            'monthly_spend_via_social_usd': {
                'bins': [0, 10, 20, 30, 50, 100, np.inf],
                'labels': [
                    '<10',
                    '10-20',
                    '20-30',
                    '30-50',
                    '50-100',
                    '>100'
                ]
            },
        },
    ),

    "Demographic & Lifestyle": dict(
        description="Who the users are: age, gender, country, profession, "
        "device and privacy settings.",
        features=[
            "age",
            "gender",
            "country",
            "profession",
            "preferred_device",
            "privacy_setting",
        ],
        category=[
            "gender",
            "country",
            "profession",
            "preferred_device",
            "privacy_setting",
        ],
        numirical=[
            "age",
        ],
        kmeans_k=4,
        kmeans_names={
            0: "Social Tablet Users",
            1: "Everyday Smartphone Users",
            2: "Marketing Professionals",
            3: "Professional Laptop/TV Users",
        },
        dbscan_eps=5.0,
        dbscan_min_samples=5,
        dbscan_names={
            -1: "Diverse Lifestyle Users",
            0: "Mainstream Lifestyle Users",
            1: "Career-Focused Users",
            2: "Independent Identity Users",
            3: "Privacy-Oriented Users",
        },
        bins={
            'age': {
                'bins': [12, 18, 21, 24, 28, 30, np.inf],
                'labels': ['12-18','18-21','21-24','24-28','28-30','>30']
            },
        },
    ),
    "Mental Health & Usage Impact": dict(
        description="Wellbeing signals: sleep disruption, self-reported "
        "mental health score, screen-time concern and daily usage.",
        features=[
            "sleep_disruption",
            "self_reported_mental_health_score",
            "screen_time_concern",
            "daily_usage_hours",
        ],
        category=[
            "sleep_disruption",
            "screen_time_concern"
        ],
        numirical=[
            "self_reported_mental_health_score",
            "daily_usage_hours"
        ],
        kmeans_k=5,
        kmeans_names={
            0: "Mid Impact Users",
            1: "Healthy Users",
            2: "High Impact Users",
            3: "Moderate Impact Users",
            4: "Balanced Users",
        },
        dbscan_eps=3.0,
        dbscan_min_samples=5,
        dbscan_names={
            -1: "Other / Noise",
            0: "Moderate Impact Users",
            1: "High Impact Users",
        },
        bins={
            'self_reported_mental_health_score': {
                'bins': [0, 2, 4, 6, 8, 10],
                'labels': [
                    'Very Low',
                    'Low',
                    'Moderate',
                    'Good',
                    'Excellent'
                ]
            },

            'daily_usage_hours': {
                'bins': [0, 1, 2, 4, 6, 8, 12, np.inf],
                'labels': [
                    '<1 hr',
                    '1-2 hrs',
                    '2-4 hrs',
                    '4-6 hrs',
                    '6-8 hrs',
                    '8-12 hrs',
                    '>12 hrs'
                ]
            },
        },
    ),
    "Platform Usage Behavior": dict(
        description="Platform-level habits: primary platform, number of "
        "platforms used, peak time, content type and privacy setting.",
        features=[
            "primary_platform",
            "platforms_used_count",
            "peak_usage_time",
            "preferred_content_type",
            "privacy_setting",
        ],
        category=[
            "primary_platform",
            "peak_usage_time",
            "preferred_content_type",
            "privacy_setting"
        ],
        numirical=[
            "platforms_used_count",
        ],
        kmeans_k=5,
        kmeans_names={
            0: "VisualUsers",
            1: "Mixed Users",
            2: "News Professionals",
            3: "Entertainment Users",
            4: "Short-Video Users",
        },
        dbscan_eps=3.5,
        dbscan_min_samples=6,
        dbscan_names={
            -1: "Diverse Platform Users",
            0: "Multi-Platform Users",
            1: "Snapchat Users",
            2: "Pinterest Users",
            3: "LinkedIn Professionals",
            4: "Video Lovers",
        },
        bins={
            'platforms_used_count': {
            'bins': [0, 1, 2, 3, 5, np.inf],
            'labels': [
                '1 Platform',
                '2 Platforms',
                '3 Platforms',
                '4-5 Platforms',
                '>5 Platforms'
            ]
             },
        },
    ),

}
CATEGORY_ICONS = {
    "User Behavior": "👤",
    "Engagement": "❤️",
    "Social Influence": "🌟",
    "Spending & Marketing": "💰",
    "Demographic & Lifestyle": "🌍",
    "Mental Health & Usage Impact": "🧠",
    "Platform Usage Behavior": "📱",
}
# BINSUSERBEHAVIOUR = {
#     'daily_usage_hours': {
#         'bins': [0, 1, 3, 6, 24],
#         'labels': ['<1 hour', '1-3 hours', '3-6 hours', '>6 hours']
#     },

#     'sessions_per_day': {
#         'bins': [0, 3, 6, 10, 50],
#         'labels': ['<3', '3-6', '6-10', '10-50']
#     },

#     'avg_session_duration_min': {
#         'bins': [0, 20, 45, 75, 300],
#         'labels': ['<20 mins', '20-45 mins', '45-75 mins', '>75 mins']
#     }
# }
REQUIRED_COLUMNS = sorted({f for cfg in CATEGORIES.values() for f in cfg["features"]})

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_kaggle_data():
    """Try to fetch the dataset via kagglehub. Returns None on failure so the
    caller can fall back to a manual file upload."""
    try:
        import kagglehub

        path = kagglehub.dataset_download(
            "C:/Users/jishn/OneDrive/Documents/DesginProject/Dataset/social_media_user_behavior.csv"
        )
        csv_candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not csv_candidates:
            return None
        return pd.read_csv(csv_candidates[0])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_local_data() -> pd.DataFrame | None:
    local_path = Path(__file__).resolve().parents[1] / "Dataset" / "social_media_user_behavior.csv"
    if local_path.exists():
        return pd.read_csv(local_path)
    return None


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    df.reset_index(drop=True, inplace=True)
    return df


def get_dataframe() -> pd.DataFrame | None:
    """Resolves the working dataframe: session cache -> kagglehub -> upload."""
    if "df_raw" in st.session_state:
        return st.session_state["df_raw"]

    with st.spinner("Loading local dataset..."):
        df = load_local_data()

    if df is not None:
        st.session_state["df_raw"] = df
        return df

    with st.spinner("Fetching dataset from Kaggle..."):
        df = load_kaggle_data()

    if df is not None:
        st.session_state["df_raw"] = df
        return df

    st.warning(
        "Couldn't auto-download the dataset from Kaggle (no network access or "
        "kagglehub isn't configured here). Please upload the CSV manually."
    )
    uploaded = st.file_uploader("Upload social_media_user_behavior.csv", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["df_raw"] = df
        return df
    return None


# ----------------------------------------------------------------------------
# Shared pipeline functions (used by every category, no per-category copies)
# ----------------------------------------------------------------------------
def preprocess_features(df: pd.DataFrame, features: list[str]):
    """One-hot encode categoricals + standard-scale everything."""
    X = df[features].copy()
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_scaled = StandardScaler().fit_transform(X_encoded)
    return X_scaled, X_encoded


@st.cache_data(show_spinner=False)
def compute_elbow(X_scaled: np.ndarray, k_max: int = 10):
    ks = list(range(2, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=500, random_state=0)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return ks, inertias


def run_kmeans(X_scaled: np.ndarray, k: int):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=500, random_state=0)
    labels = km.fit_predict(X_scaled)
    return km, labels


def run_dbscan(X_scaled: np.ndarray, eps: float, min_samples: int):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return db, labels


def pca_2d(X_scaled: np.ndarray, labels: np.ndarray, name_map: dict | None = None):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_scaled)
    pdf = pd.DataFrame(coords, columns=["PC1", "PC2"])
    if name_map:
        pdf["Segment"] = pd.Series(labels).map(lambda x: name_map.get(x, f"Cluster {x}")).values
    else:
        pdf["Segment"] = pd.Series(labels).astype(str).values
    return pdf, pca.explained_variance_ratio_


def plot_pca(pdf: pd.DataFrame, title: str):
    fig = px.scatter(
        pdf,
        x="PC1",
        y="PC2",
        color="Segment",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        height=560,
        title_x=0.5,
        legend_title="Segment",
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
    )
    return fig


def cluster_profile(df: pd.DataFrame, cluster_col: str, features: list[str]):
    numeric_feats = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    cat_feats = [f for f in features if f not in numeric_feats]

    numeric_summary = None
    if numeric_feats:
        numeric_summary = df.groupby(cluster_col)[numeric_feats].mean().round(2)

    cat_summary = None
    if cat_feats:
        cat_summary = pd.DataFrame(
            {
                f: df.groupby(cluster_col)[f].agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"
                )
                for f in cat_feats
            }
        )
    return numeric_summary, cat_summary


def safe_metrics(X_scaled: np.ndarray, labels: np.ndarray):
    """Silhouette / Davies-Bouldin, ignoring DBSCAN noise (-1) points and
    returning (None, None) when fewer than 2 real clusters exist."""
    labels = np.asarray(labels)
    real_clusters = set(labels) - {-1}
    if len(real_clusters) < 2:
        return None, None
    mask = labels != -1
    try:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
        return sil, dbi
    except Exception:
        return None, None


def comparison_chart(sil_k, dbi_k, sil_d, dbi_d):
    scores_df = pd.DataFrame(
        {
            "Algorithm": ["K-Means", "DBSCAN"],
            "Silhouette Score": [sil_k, sil_d],
            "Davies-Bouldin Index": [dbi_k, dbi_d],
        }
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Silhouette Score (Higher is Better)",
            "Davies-Bouldin Index (Lower is Better)",
        ),
    )

    fig_sil = px.bar(
        scores_df,
        x="Algorithm",
        y="Silhouette Score",
        color="Algorithm",
        color_discrete_sequence=["#636EFA", "#EF553B"],
        text_auto=".2f",
        template="plotly",      # <-- adaptive
    )

    fig_dbi = px.scatter(
        scores_df,
        x="Algorithm",
        y="Davies-Bouldin Index",
        color="Algorithm",
        color_discrete_sequence=["#00CC96", "#AB63FA"],
        text="Davies-Bouldin Index",
        template="plotly",      # <-- adaptive
    )

    for trace in fig_sil.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig_dbi.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_traces(
        selector=dict(type="bar"),
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_traces(
        selector=dict(type="scatter"),
        mode="markers+text",
        textposition="top center",
        marker=dict(size=16, line=dict(width=0)),
    )

    fig.update_layout(
        height=460,
        showlegend=False,
        title_x=0.5,
        margin=dict(t=70, b=40, l=40, r=40),
        template="plotly",              # <-- adaptive
        plot_bgcolor="rgba(0,0,0,0)",   # transparent
        paper_bgcolor="rgba(0,0,0,0)",  # transparent
        font=dict(size=13),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(150,150,150,0.25)",  # visible in both themes
        zeroline=False,
    )

    return fig


def feature_distribution_figs(df: pd.DataFrame, features: list[str]):
    """Return a list of (feature, plotly figure) for a quick EDA look."""
    figs = []

    for f in features:

        if pd.api.types.is_numeric_dtype(df[f]):

            fig = px.histogram(
                df,
                x=f,
                nbins=30,
                title=f.replace("_", " ").title(),
                color_discrete_sequence=px.colors.qualitative.Safe,
                template="plotly",
                opacity=0.9,
            )

            fig.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            )

        else:

            counts = df[f].value_counts().reset_index()
            counts.columns = [f, "Count"]

            fig = px.bar(
                counts,
                x=f,
                y="Count",
                title=f.replace("_", " ").title(),
               color_discrete_sequence=px.colors.qualitative.Safe,
                template="plotly",
                text_auto=True,
            )

            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
            )

        fig.update_layout(
            height=300,
            title_x=0.5,
            template="plotly",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=30, l=20, r=20),
            xaxis_title=None,
            yaxis_title="Count",
            showlegend=False,
            font=dict(size=13),
        )

        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(150,150,150,0.20)",
            zeroline=False,
        )

        figs.append((f, fig))

    return figs

def features_categorical(features, df, cluster_col):

    figs = []

    for feature in features:

        fig = px.histogram(
            df,
            x=cluster_col,
            color=feature,
            width=800,
            height=500,
            barmode="group",
            text_auto=True,
            template="plotly_white",
            title=feature.replace("_", " ").title(),
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title="Cluster",
            yaxis_title="Number of Users",
            legend_title=feature.replace("_", " ").title(),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.30,
                xanchor="right",
                x=1,
                title=None
            ),
            margin=dict(
                t=50,
                b=120
            )
        )

        figs.append((feature, fig))

    return figs

def features_numerical(category_config, cluster_col, df):

    figs = []

    bins_config = category_config.get("bins", {})
    numerical_features = list(bins_config.keys())

    for feature in numerical_features:

        config = bins_config.get(feature)

        if not config:
            continue

        temp_col = f"{feature}_group"

        # Convert numerical values into bins
        df[temp_col] = pd.cut(
            df[feature],
            bins=config["bins"],
            labels=config["labels"],
            include_lowest=True,
            ordered=True
        )

        # Count users in each bin for every cluster
        count_df = (
            df.groupby([temp_col, cluster_col], observed=False)
              .size()
              .reset_index(name="Count")
        )

        # Create grouped bar chart
        fig = px.bar(
            count_df,
            x=temp_col,
            y="Count",
            color=cluster_col,
            barmode="group",
            text="Count",
            width=800,height=500,
            title=f"{feature.replace('_', ' ').title()}",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=450,
            title_x=0.5,
            xaxis_title=feature.replace("_", " ").title(),
            yaxis_title="Number of Users",
            legend_title="Cluster",
            bargap=0.20,
            bargroupgap=0.05,
            xaxis=dict(
                categoryorder="array",
                categoryarray=config["labels"],
                tickangle=0
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.50,
                xanchor="right",
                x=1,
                title=None
            )
        )

        figs.append((feature, fig))

        # Remove temporary column
        df.drop(columns=[temp_col], inplace=True)

    return figs


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("☰ Navigation")
# page = st.sidebar.radio("Go to", ["🏠 Dataset Overview"] + [f"🔎 {c}" for c in CATEGORIES])
pages = ["🏠 Dataset Overview"] + [
    f"{CATEGORY_ICONS.get(category, '📂')} {category}"
    for category in CATEGORIES
]

page = st.sidebar.radio("Go to", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔄 Analysis Workflow

- 📋 Feature Preparation
- 📐 Elbow Method
- 🤖 K-Means Clustering
- 🔍 DBSCAN Clustering
- 📉 PCA Visualization
- 👥 Cluster Profiling
- 📊 Algorithm Comparison
""")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("Social Media User Behavior — Unsupervised Learning Dashboard")

df_raw = get_dataframe()
if df_raw is None:
    st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
if missing:
    st.error(f"The uploaded dataset is missing expected columns: {missing}")
    st.stop()

df = clean_data(df_raw)

# ---- Dataset Overview page ----
if page == "🏠 Dataset Overview":
    st.subheader("Dataset snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Categories available", len(CATEGORIES))

    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Column summary (describe)"):
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("### Categories in this dashboard")
    for name, cfg in CATEGORIES.items():
        st.markdown(f"**{name}** — {cfg['description']}")
        st.caption(", ".join(cfg["features"]))

    st.info("Pick a category from the sidebar to explore its clustering pipeline.")

# ---- Category pages ----
else:
    category = page.split(" ", 1)[1]
    cfg = CATEGORIES[category]
    features = cfg["features"]
    numirical_features = cfg["numirical"]
    category_features = cfg["category"]
    bins_config = cfg["bins"]

    st.subheader(category)
    st.caption(cfg["description"])
    st.write(f"**Features used:** {', '.join(features)}")

    X_scaled, X_encoded = preprocess_features(df, features)

    tab_overview, tab_kmeans, tab_dbscan, tab_compare, tab_patterns,tab_patterns_dbscan = st.tabs(
        ["Feature Overview", "KMeans Clustering", "DBSCAN Clustering", "Algorithm Comparison", "Patterns Comparison by KMeans", "Patterns Comparison by DBSCAN"]
    )

    # --- Feature overview ---
    with tab_overview:
        st.markdown("#### Raw feature distributions (before clustering)")
        figs = feature_distribution_figs(df, features)
        cols = st.columns(2)
        for i, (feat, fig) in enumerate(figs):
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Elbow method (optimal K)")
        with st.spinner("Computing inertia for k = 2..10..."):
            ks, inertias = compute_elbow(X_scaled)
        elbow_fig = px.line(x=ks, y=inertias, markers=True,
                             labels={"x": "Number of Clusters (K)", "y": "Inertia"},
                             title="Elbow Method")
        elbow_fig.update_layout(height=400, title_x=0.5)
        st.plotly_chart(elbow_fig, use_container_width=True)
        st.caption(f"This category was profiled with K = {cfg['kmeans_k']} based on the elbow point and cluster interpretability.")

    # --- KMeans ---
    with tab_kmeans:
        k = st.slider("Number of clusters (K)", min_value=2, max_value=10,
                      value=cfg["kmeans_k"], key=f"k_{category}")
        kmeans_model, kmeans_labels = run_kmeans(X_scaled, k)
        df[f"_kmeans_{category}"] = kmeans_labels

        name_map = cfg["kmeans_names"] if k == cfg["kmeans_k"] else None
        if name_map is None:
            st.info("Custom K selected — showing generic cluster numbers instead of the named segments.")

        pca_df, var_ratio = pca_2d(X_scaled, kmeans_labels, name_map)
        st.plotly_chart(plot_pca(pca_df, f"PCA — {category} (KMeans)"), use_container_width=True)
        st.caption(f"Explained variance: PC1 = {var_ratio[0]*100:.1f}%, PC2 = {var_ratio[1]*100:.1f}%")

        st.markdown("#### Cluster sizes")
        label_col = pca_df["Segment"]
        st.plotly_chart(
            px.bar(
                label_col.value_counts().reset_index(),
                x="Segment",
                y="count",
                text="count",
                color="Segment",
                color_discrete_sequence=px.colors.qualitative.Dark24,
            ).update_layout(
                xaxis_title="Cluster",
                yaxis_title="Number of Users",
                xaxis_tickangle=0,
                showlegend=False,
                title="Cluster Distribution",
                title_x=0.5,
            ),
            use_container_width=True,
        )
        st.markdown("#### Cluster profile")
        df["_profile_col"] = label_col.values
        numeric_summary, cat_summary = cluster_profile(df, "_profile_col", features) # CLuster profiling for KMeans

        # ---------------- Cluster Profile ----------------
        st.markdown("### Cluster Profile Summary")

        if numeric_summary is not None:
            st.markdown("#### Average Numeric Features")

            styled_numeric = (
                numeric_summary.style
                .format("{:.2f}")
                .background_gradient(cmap="Blues", axis=0)
                .highlight_max(color="#90EE90", axis=0)   # Highest value
                .highlight_min(color="#FFCCCB", axis=0)   # Lowest value
                .set_properties(**{
                    "text-align": "center",
                    "font-size": "14px",
                    "font-weqight": "bold"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#1F4E79"),
                            ("color", "white"),
                            ("font-weight", "bold"),
                            ("text-align", "center"),
                            ("padding", "8px")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("padding", "6px")
                        ]
                    }
                ])
            )

            st.dataframe(styled_numeric, use_container_width=True)

        if cat_summary is not None:
            st.markdown("#### Dominant Categorical Features")

            styled_cat = (
                cat_summary.style
                .set_properties(**{
                    "text-align": "center",
                    "font-size": "14px",
                    "font-weqight": "bold"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#2E8B57"),
                            ("color", "white"),
                            ("font-weight", "bold"),
                            ("text-align", "center"),
                            ("padding", "8px")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("padding", "6px")
                        ]
                    }
                ])
            )

            st.dataframe(styled_cat, use_container_width=True)

    # --- DBSCAN ---
    with tab_dbscan:
        c1, c2 = st.columns(2)
        eps = c1.number_input("eps", min_value=0.1, value=float(cfg["dbscan_eps"]), step=0.1, key=f"eps_{category}")
        min_samples = c2.number_input("min_samples", min_value=2, value=int(cfg["dbscan_min_samples"]), step=1, key=f"ms_{category}")

        _, dbscan_labels = run_dbscan(X_scaled, eps, min_samples)
        n_noise = int((dbscan_labels == -1).sum())
        st.caption(f"Noise points (label -1): {n_noise} of {len(dbscan_labels)}")

        is_default_params = (eps == cfg["dbscan_eps"]) and (min_samples == cfg["dbscan_min_samples"])
        name_map = cfg["dbscan_names"] if is_default_params else None
        if name_map is None:
            st.info("Custom eps/min_samples — showing generic cluster numbers instead of the named segments.")

        pca_df_db, var_ratio_db = pca_2d(X_scaled, dbscan_labels, name_map)
        st.plotly_chart(plot_pca(pca_df_db, f"PCA — {category} (DBSCAN)"), use_container_width=True)
        st.caption(f"Explained variance: PC1 = {var_ratio_db[0]*100:.1f}%, PC2 = {var_ratio_db[1]*100:.1f}%")

        st.markdown("#### Cluster sizes")
        # st.bar_chart(pca_df_db["Segment"].value_counts())
        label_col = pca_df_db["Segment"]
        st.plotly_chart(
            px.bar(
                label_col.value_counts().reset_index(),
                x="Segment",
                y="count",
                text="count",
                color="Segment",
                color_discrete_sequence=px.colors.qualitative.Dark24,
            ).update_layout(
                xaxis_title="Cluster",
                yaxis_title="Number of Users",
                xaxis_tickangle=0,
                showlegend=False,
                title="Cluster Distribution",
                title_x=0.5,
            ),
            use_container_width=True,
        )

        st.markdown("#### Cluster profile")
        df["_profile_col_db"] = pca_df_db["Segment"].values
        numeric_summary_db, cat_summary_db = cluster_profile(df, "_profile_col_db", features) # Cluster profiling for DBSCAN

        if numeric_summary_db is not None:
            st.markdown("**Average of numeric features per cluster**")
            st.dataframe(numeric_summary_db, use_container_width=True)
        if cat_summary_db is not None:
            st.markdown("**Most common category per cluster**")
            st.dataframe(cat_summary_db, use_container_width=True)

    # --- Comparison ---
    with tab_compare:
        sil_k, dbi_k = safe_metrics(X_scaled, kmeans_labels)
        sil_d, dbi_d = safe_metrics(X_scaled, dbscan_labels)

        c1, c2 = st.columns(2)
        c1.metric("KMeans Silhouette", f"{sil_k:.2f}" if sil_k is not None else "n/a")
        c1.metric("KMeans Davies-Bouldin", f"{dbi_k:.2f}" if dbi_k is not None else "n/a")
        c2.metric("DBSCAN Silhouette", f"{sil_d:.2f}" if sil_d is not None else "n/a")
        c2.metric("DBSCAN Davies-Bouldin", f"{dbi_d:.2f}" if dbi_d is not None else "n/a")

        if None not in (sil_k, dbi_k, sil_d, dbi_d):
            st.plotly_chart(comparison_chart(sil_k, dbi_k, sil_d, dbi_d), use_container_width=True)
        else:
            st.warning(
                "Not enough valid clusters to compute a full comparison "
                "(try adjusting DBSCAN's eps/min_samples in the previous tab)."
            )

    #--- Patterns Comparison ---
    with tab_patterns:
        # Categorical Feature Comparison By KMeans
        st.subheader("Categorical Feature Pattens BY Kmeans")
        st.caption("This section compares the distribution of categorical features across KMeans ")

        figures = features_categorical(
            category_features,
            df,
            "_profile_col"
        )

        for i in range(0, len(figures), 2):

            col1, col2 = st.columns(2)

            feature, fig = figures[i]
            with col1:
                st.plotly_chart(fig, use_container_width=True)

            if i + 1 < len(figures):
                feature, fig = figures[i + 1]
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

        # Numirical features Comparison by kmeans 
        st.subheader("Numirical Features Pattens By Kmeans")
        st.caption("This section compares the distribution of numirical features across KMeans ")
        figures_numirical = features_numerical(
            cfg,
            "_profile_col",
            df

        )

        for i in range(0,len(figures_numirical),2):
            
            col1, col2 = st.columns(2)

            feature, fig = figures_numirical[i]
            with col1:
                st.plotly_chart(fig, use_container_width=True)

            if i + 1 < len(figures_numirical):
                feature,fig = figures_numirical[i + 1]
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

    with tab_patterns_dbscan:
        # Categorical Feature Comparison By DBSCAN
        st.subheader("Categorical Feature Pattens BY DBSCAN")
        st.caption("This section compares the distribution of categorical features across DBSCAN ")

        figures = features_categorical(
            category_features,
            df,
            "_profile_col_db"
        )

        for i in range(0, len(figures), 2):

            col1, col2 = st.columns(2)

            feature, fig = figures[i]
            with col1:
                st.plotly_chart(fig, use_container_width=True)

            if i + 1 < len(figures):
                feature, fig = figures[i + 1]
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

        # Numirical features Comparison by DBSCAN 
        st.subheader("Numirical Features Pattens By DBSCAN")
        st.caption("This section compares the distribution of numirical features across DBSCAN ")
        figures_numirical = features_numerical(
            cfg,
            "_profile_col_db",
            df

        )

        for i in range(0,len(figures_numirical),2):
            
            col1, col2 = st.columns(2)

            feature, fig = figures_numirical[i]
            with col1:
                st.plotly_chart(fig, use_container_width=True)

            if i + 1 < len(figures_numirical):
                feature,fig = figures_numirical[i + 1]
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

    # clean up scratch columns so re-runs stay tidy
    for tmp_col in [f"_kmeans_{category}", "_profile_col", "_profile_col_db"]:
        if tmp_col in df.columns:
            df.drop(columns=[tmp_col], inplace=True)
