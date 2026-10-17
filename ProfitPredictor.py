
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("polynomial_model1.pkl", "rb") as f:
    model = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title="Prediction App",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Prediction App")
st.write("Enter the details below to generate a prediction.")

# User input form
with st.form("prediction_form"):

    st.subheader("Numerical Inputs")

    creator_followers_k = st.number_input(
        "Creator Followers (K)",
        min_value=0.0,
        value=200.0,
        step=10.0
    )

    engagement_rate_pct = st.number_input(
        "Engagement Rate (%)",
        min_value=0.0,
        value=4.5,
        step=0.1
    )

    paid_boost_spend_usd = st.number_input(
        "Paid Boost Spend (USD)",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

    avg_watch_time_sec = st.number_input(
        "Average Watch Time (seconds)",
        min_value=0.0,
        value=25.0,
        step=1.0
    )

    promo_discount_pct = st.number_input(
        "Promo Discount (%)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    caption_word_count = st.number_input(
        "Caption Word Count",
        min_value=0,
        value=100,
        step=5
    )

    hashtags_used_count = st.number_input(
        "Hashtags Used Count",
        min_value=0,
        value=8,
        step=1
    )

    st.subheader("Categorical Inputs")

    content_format = st.selectbox(
        "Content Format",
        ["Video", "Image", "Carousel", "Story", "Reel"]
    )

    audience_intent_tier = st.selectbox(
        "Audience Intent Tier",
        ["Low", "Medium", "High"]
    )

    creator_phone_os = st.selectbox(
        "Creator Phone OS",
        ["iOS", "Android", "Other"]
    )

    contract_payment_terms = st.selectbox(
        "Contract Payment Terms",
        ["Fixed", "Performance-Based", "Hybrid"]
    )

    posting_day_weather = st.selectbox(
        "Posting Day Weather",
        ["Sunny", "Cloudy", "Rainy", "Windy", "Snowy"]
    )

    # Prediction button
    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

# Make prediction
if submitted:

    custom_input = pd.DataFrame([{
        "creator_followers_k": creator_followers_k,
        "engagement_rate_pct": engagement_rate_pct,
        "paid_boost_spend_usd": paid_boost_spend_usd,
        "avg_watch_time_sec": avg_watch_time_sec,
        "promo_discount_pct": promo_discount_pct,
        "content_format": content_format,
        "audience_intent_tier": audience_intent_tier,
        "caption_word_count": caption_word_count,
        "hashtags_used_count": hashtags_used_count,
        "creator_phone_os": creator_phone_os,
        "contract_payment_terms": contract_payment_terms,
        "posting_day_weather": posting_day_weather
    }])

    try:
        prediction = model.predict(custom_input)

        st.success("Prediction completed successfully!")

        st.subheader("Predicted Output")
        st.metric(
            label="Prediction",
            value=f"{prediction[0]:,.2f}"
        )

        with st.expander("View Input Data"):
            st.dataframe(custom_input, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the model includes the same preprocessing "
            "and feature transformations used during training."
        )