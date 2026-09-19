import os
import sys
import sqlite3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "tourism_master_cleaned.csv"
)

CLASSIFIER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "visitmode_classification_deployment.pkl"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "visitmode_classes.pkl"
)

REGRESSION_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rating_prediction_model.pkl"
)

RECOMMENDATION_PATH = os.path.join(
    BASE_DIR,
    "models",
    "recommendation",
    "recommendation_artifacts.pkl"
)


@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


SQL_PATH = os.path.join(
    BASE_DIR,
    "sql",
    "tourism_analysis.sql"
)


@st.cache_resource
def load_sql_connection():

    connection = sqlite3.connect(
        ":memory:",
        check_same_thread=False
    )

    sql_data = load_data().rename(
        columns={
            "transactionid": "TransactionId",
            "userid": "UserId",
            "visityear": "VisitYear",
            "visitmonth": "VisitMonth",
            "visitmode": "VisitMode",
            "attractionid": "AttractionId",
            "rating": "Rating",
            "attractiontypeid": "AttractionTypeId"
        }
    )

    sql_data.to_sql(
        "Transaction",
        connection,
        index=False,
        if_exists="replace"
    )

    item_data = sql_data[
        ["AttractionId", "AttractionTypeId"]
    ].drop_duplicates()

    item_data.to_sql(
        "Item",
        connection,
        index=False,
        if_exists="replace"
    )

    return connection


@st.cache_data
def load_sql_queries():

    sql_text = Path(SQL_PATH).read_text(
        encoding="utf-8"
    )

    return [
        query.strip()
        for query in sql_text.split(";")
        if query.strip()
    ]
    
@st.cache_resource
def load_classifier():

    return joblib.load(
        CLASSIFIER_PATH
    )

@st.cache_resource
def load_classes():

    return joblib.load(
        CLASSES_PATH
    )

@st.cache_resource
def load_regression_model():

    return joblib.load(
        REGRESSION_PATH
    )

@st.cache_resource
def load_recommendation_artifacts():

    return joblib.load(
        RECOMMENDATION_PATH
    )

try:

    df = load_data()

    classifier = load_classifier()

    classes = load_classes()

    regression_model = load_regression_model()

    recommendation_artifacts = (
        load_recommendation_artifacts()
    )

except Exception as e:

    st.error(
        "Application could not load the required data/models."
    )

    st.exception(e)

    st.stop()
    
user_item_matrix = (
    recommendation_artifacts[
        "user_item_matrix"
    ]
)

eval_matrix = (
    recommendation_artifacts[
        "eval_matrix"
    ]
)

eval_item_index = {
    item_id: index
    for index, item_id in enumerate(
        eval_matrix.columns
    )
}

item_similarity = (
    recommendation_artifacts[
        "item_similarity"
    ]
)

eval_item_similarity = (
    recommendation_artifacts[
        "eval_item_similarity"
    ]
)

content_similarity = (
    recommendation_artifacts[
        "content_similarity"
    ]
)

content_item_ids = (
    recommendation_artifacts[
        "content_item_ids"
    ]
)

content_item_index = (
    recommendation_artifacts[
        "content_item_index"
    ]
)

attraction_lookup = (
    recommendation_artifacts[
        "attraction_lookup"
    ]
)

popular_items = (
    recommendation_artifacts[
        "popular_items"
    ]
)

st.title(
    "🌍 Tourism Experience Analytics"
)

st.markdown(
    """
    ### AI-Powered Tourism Intelligence Platform

    Explore tourism trends, predict visitor behavior,
    estimate attraction ratings, and discover
    personalized attractions.
    """
)

st.sidebar.title(
    "🧭 Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "📊 Dashboard",
        "🗃️ SQL Analytics",
        "🤖 Visit Mode Prediction",
        "⭐ Rating Prediction",
        "🎯 Recommendations"
    ]
)

if page == "📊 Dashboard":

    st.header(
        "📊 Tourism Analytics Dashboard"
    )
    
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions",
        f"{len(df):,}"
    )

    col2.metric(
        "Users",
        f"{df['userid'].nunique():,}"
    )

    col3.metric(
        "Attractions",
        f"{df['attractionid'].nunique():,}"
    )

    col4.metric(
        "Average Rating",
        f"{df['rating'].mean():.2f} / 5"
    )

    mode_counts = (
        df["visitmode"]
        .value_counts()
        .reset_index()
    )

    mode_counts.columns = [
        "VisitMode",
        "Count"
    ]

    fig = px.bar(
        mode_counts,
        x="VisitMode",
        y="Count",
        title="Visitor Mode Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_counts.columns = [
        "Rating",
        "Count"
    ]

    fig = px.bar(
        rating_counts,
        x="Rating",
        y="Count",
        title="Rating Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )   
    
    popular_attractions = (
        df.groupby("attraction")
        .agg(
            Visits=("userid", "count"),
            Average_Rating=("rating", "mean")
        )
        .reset_index()
        .sort_values(
            "Visits",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        popular_attractions,
        x="Visits",
        y="attraction",
        orientation="h",
        title="Top 10 Most Visited Attractions"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    yearly = (
        df.groupby("visityear")
        .agg(
            Transactions=("transactionid", "count"),
            Average_Rating=("rating", "mean")
        )
        .reset_index()
    )

    fig = px.line(
        yearly,
        x="visityear",
        y="Transactions",
        markers=True,
        title="Tourism Transactions by Year"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
elif page == "🗃️ SQL Analytics":

    st.header(
        "🗃️ SQL Tourism Analytics"
    )

    st.write(
        "Run the validated queries from "
        "sql/tourism_analysis.sql against the "
        "processed tourism dataset."
    )

    sql_operation_names = [
        "Total transactions",
        "Total users",
        "Total attractions",
        "Average rating",
        "Rating distribution",
        "Visit mode distribution",
        "Transactions by year",
        "Transactions by month",
        "Top 10 attractions",
        "Attraction ratings",
        "Rating by visit mode",
        "Visit mode and rating",
        "Attraction performance",
        "Highly rated attractions",
        "Attractions with 100+ visits",
        "Attractions with 500+ visits and rating 4+",
        "User activity",
        "Most active users",
        "User rating behavior",
        "Attraction type analysis",
        "Visit mode by year",
        "Average rating by year"
    ]

    selected_operation = st.selectbox(
        "Select SQL operation",
        sql_operation_names
    )

    sql_queries = load_sql_queries()
    query_index = sql_operation_names.index(
        selected_operation
    )
    sql_connection = load_sql_connection()
    sql_result = pd.read_sql_query(
        sql_queries[query_index],
        sql_connection
    )

    st.metric(
        "Rows returned",
        f"{len(sql_result):,}"
    )
    st.dataframe(
        sql_result,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download results as CSV",
        sql_result.to_csv(index=False),
        file_name="tourism_sql_results.csv",
        mime="text/csv",
        use_container_width=True
    )

elif page == "🤖 Visit Mode Prediction":

    st.header(
        "🤖 Visitor Mode Prediction"
    )

    st.write(
        "Enter visitor and trip information "
        "to predict the likely VisitMode."
    )
    
    col1, col2 = st.columns(2)

    with col1:

        continent = st.selectbox(
            "User Continent",
            sorted(
                df["user_continent"]
                .dropna()
                .unique()
            )
        )

        region = st.selectbox(
            "User Region",
            sorted(
                df["user_region"]
                .dropna()
                .unique()
            )
        )

        country = st.selectbox(
            "User Country",
            sorted(
                df["user_country"]
                .dropna()
                .unique()
            )
        )

    with col2:

        year = st.number_input(
            "Visit Year",
            min_value=int(
                df["visityear"].min()
            ),
            max_value=int(
                df["visityear"].max()
            ),
            value=int(
                df["visityear"].max()
            )
        )

        month = st.selectbox(
            "Visit Month",
            list(range(1, 13))
        )

        rating = st.slider(
            "Expected Rating",
            min_value=1,
            max_value=5,
            value=4
        )
        
elif page == "⭐ Rating Prediction":

    st.header(
        "⭐ Attraction Rating Prediction"
    )

    st.write(
        "Estimate the expected attraction rating "
        "for a tourism interaction."
    )
    
    col1, col2 = st.columns(2)

    with col1:

        selected_attraction = st.selectbox(
            "Attraction",
            sorted(
                df["attraction"]
                .dropna()
                .unique()
            )
        )

        selected_type = st.selectbox(
            "Attraction Type",
            sorted(
                df["attractiontype"]
                .dropna()
                .unique()
            )
        )

    with col2:

        selected_year = st.number_input(
            "Visit Year",
            min_value=int(
                df["visityear"].min()
            ),
            max_value=int(
                df["visityear"].max()
            ),
            value=int(
                df["visityear"].max()
            )
        )

        selected_month = st.selectbox(
            "Visit Month",
            list(range(1, 13))
        )
        
elif page == "🎯 Recommendations":

    st.header(
        "🎯 Personalized Attraction Recommendations"
    )

    st.write(
        "Enter a User ID to generate "
        "personalized attraction recommendations."
    )

    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=14,
        step=1
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )
    
def streamlit_recommendations(
    user_id,
    top_n=5
):

    if user_id not in eval_matrix.index:

        return (
            popular_items[
                [
                    "attractionid",
                    "weighted_score"
                ]
            ]
            .head(top_n)
            .rename(
                columns={
                    "weighted_score": "Hybrid Score"
                }
            )
            .merge(
                attraction_lookup.reset_index(),
                on="attractionid",
                how="left"
            )
        )

    user_ratings = eval_matrix.loc[
        user_id
    ]

    liked_items = user_ratings[
        user_ratings >= 4
    ].dropna()

    collaborative_scores = np.zeros(
        len(content_item_ids)
    )

    content_scores = np.zeros(
        len(content_item_ids)
    )

    # Collaborative
    for item_id, rating in liked_items.items():

        if item_id not in content_item_index:
            continue

        if item_id not in eval_item_index:
            continue

        eval_idx = eval_item_index[item_id]

        content_idx = content_item_index[
            item_id
        ]

        collaborative_scores += (
            eval_item_similarity[eval_idx]
            * float(rating)
        )

    # Content
    total_weight = 0

    for item_id, rating in liked_items.items():

        if item_id not in content_item_index:
            continue

        idx = content_item_index[item_id]

        content_scores += (
            content_similarity[idx]
            * float(rating)
        )

        total_weight += float(rating)

    if total_weight > 0:

        content_scores /= total_weight

    # Popularity
    popularity_map = dict(
        zip(
            popular_items["attractionid"],
            popular_items["weighted_score"]
        )
    )

    popularity_scores = np.array([
        popularity_map.get(
            item_id,
            0
        )
        for item_id in content_item_ids
    ])

    # Normalize
    def normalize(x):

        minimum = x.min()
        maximum = x.max()

        if maximum == minimum:
            return np.zeros_like(x)

        return (
            (x - minimum)
            /
            (maximum - minimum)
        )

    collaborative_scores = normalize(
        collaborative_scores
    )

    content_scores = normalize(
        content_scores
    )

    popularity_scores = normalize(
        popularity_scores
    )

    # Final hybrid score
    hybrid_scores = (
        0.70 * collaborative_scores
        +
        0.20 * content_scores
        +
        0.10 * popularity_scores
    )

    # Remove already visited attractions
    seen_items = user_ratings.dropna().index

    for item_id in seen_items:

        if item_id in content_item_index:

            hybrid_scores[
                content_item_index[item_id]
            ] = -np.inf

    top_indices = np.argsort(
        hybrid_scores
    )[::-1]

    recommendations = []

    for idx in top_indices:

        if len(recommendations) >= top_n:
            break

        if hybrid_scores[idx] == -np.inf:
            continue

        recommendations.append({
            "attractionid":
                content_item_ids[idx],
            "Hybrid Score":
                hybrid_scores[idx]
        })

    result = pd.DataFrame(
        recommendations
    )

    result = result.merge(
        attraction_lookup.reset_index(),
        on="attractionid",
        how="left"
    )

    return result

if page == "🎯 Recommendations" and st.button(
    "🔍 Generate Recommendations",
    use_container_width=True
):

    recommendations = (
        streamlit_recommendations(
            user_id,
            top_n
        )
    )

    st.subheader(
        "🌟 Recommended Attractions"
    )

    for i, row in recommendations.iterrows():

        st.markdown(
            f"""
            ### {i + 1}. {row['attraction']}

            **Type:** {row['attractiontype']}

            **Recommendation Score:** \
            {row['Hybrid Score']:.3f}
            """
        )

        st.divider()

if page == "🤖 Visit Mode Prediction" and st.button(
    "🔮 Predict Visit Mode",
    use_container_width=True
):

    matching_rows = df[
        (df["user_continent"] == continent)
        & (df["user_region"] == region)
        & (df["user_country"] == country)
    ]

    sample_row = (
        matching_rows.iloc[0]
        if not matching_rows.empty
        else df.iloc[0]
    )

    prediction_input = pd.DataFrame([{
        feature: sample_row[feature]
        if feature in sample_row.index
        else 0
        for feature in classifier.feature_names_in_
    }])

    prediction_input["visityear"] = year
    prediction_input["visitmonth"] = month

    predicted_mode = classifier.predict(
        prediction_input
    )[0]

    st.success(
        f"Predicted visit mode: {predicted_mode}"
    )

if page == "⭐ Rating Prediction" and st.button(
    "⭐ Predict Rating",
    use_container_width=True
):

    matching_rows = df[
        (df["attraction"] == selected_attraction)
        & (df["attractiontype"] == selected_type)
    ]

    sample_row = (
        matching_rows.iloc[0]
        if not matching_rows.empty
        else df.iloc[0]
    )

    prediction_input = pd.DataFrame([{
        feature: sample_row[feature]
        if feature in sample_row.index
        else 0
        for feature in regression_model.feature_names_in_
    }])

    prediction_input["visityear"] = selected_year
    prediction_input["visitmonth"] = selected_month

    predicted_rating = float(
        regression_model.predict(prediction_input)[0]
    )

    st.success(
        f"Predicted attraction rating: "
        f"{np.clip(predicted_rating, 1, 5):.2f} / 5"
    )
            
 
