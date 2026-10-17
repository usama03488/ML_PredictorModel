import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Return Risk Prediction App", page_icon="📦", layout="centered")

st.title("📦 E-Commerce Return Risk Prediction")
st.write("Enter the customer and order details below to predict the likelihood of a return.")

# --- Load the Saved Pipeline Model with Version Patch ---
@st.cache_resource
def load_model():
    with open("Lg_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Compatibility Patch: Fix scikit-learn version drift for predict_proba
    classifier = model.named_steps['classifier'] if hasattr(model, 'named_steps') else model
    if not hasattr(classifier, 'multi_class'):
        classifier.multi_class = 'auto'
        
    return model

pipeline = load_model()

# --- User Input Form ---
with st.form("prediction_form"):
    st.subheader("Order & Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cart_value_usd = st.number_input("Cart Value (USD)", min_value=0.0, max_value=5000.0, value=172.17, step=1.0)
        past_return_rate = st.slider("Past Return Rate", min_value=0.0, max_value=1.0, value=0.233, step=0.01)
        size_bracketing_flag = st.selectbox("Size Bracketing Flag", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        delivery_days_taken = st.number_input("Delivery Days Taken", min_value=1, max_value=30, value=11, step=1)
        discount_pct_applied = st.slider("Discount Pct Applied (%)", min_value=0, max_value=100, value=10, step=1)
        newsletter_subscriber = st.selectbox("Newsletter Subscriber", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        product_category = st.selectbox("Product Category", options=["Footwear", "Accessories", "Formal-Eveningwear", "Casual-Tops"])
        fit_review_sentiment = st.selectbox("Fit Review Sentiment", options=["True-to-Size", "Inconsistent-Sizing", "Runs-Slightly-Small"])
        browser_used = st.selectbox("Browser Used", options=["Safari", "Firefox", "Edge", "Chrome"])
        checkout_device = st.selectbox("Checkout Device", options=["Android-App", "iOS-App", "Desktop-Web"])
        packaging_color = st.selectbox("Packaging Color", options=["Kraft-Brown", "Branded-Teal", "Minimal-White"])
        loyalty_card_color = st.selectbox("Loyalty Card Color", options=["Platinum", "Silver", "Gold"])

    submitted = st.form_submit_button("Predict Return Risk")

# --- Prediction Logic ---
if submitted:
    input_data = pd.DataFrame({
        "cart_value_usd": [cart_value_usd],
        "past_return_rate": [past_return_rate],
        "size_bracketing_flag": [int(size_bracketing_flag)],
        "delivery_days_taken": [int(delivery_days_taken)],
        "discount_pct_applied": [int(discount_pct_applied)],
        "newsletter_subscriber": [int(newsletter_subscriber)],
        "product_category": [product_category],
        "fit_review_sentiment": [fit_review_sentiment],
        "browser_used": [browser_used],
        "checkout_device": [checkout_device],
        "packaging_color": [packaging_color],
        "loyalty_card_color": [loyalty_card_color]
    })

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Return Risk** (Probability: {probability[1]*100:.2f}%)")
    else:
        st.success(f"✅ **Low Return Risk** (Probability of keeping: {probability[0]*100:.2f}%)")