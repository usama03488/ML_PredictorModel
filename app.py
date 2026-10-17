import streamlit as st

st.set_page_config(
    page_title="Intelligence Hub", 
    page_icon="📊", 
    layout="centered"
)

st.title("📊 E-Commerce & Campaign Intelligence Hub")
st.write("Choose a tool below to open its prediction page:")

st.markdown("---")

# Create two columns for a clean layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Customer Return Risk")
    st.write("Predict whether a customer is likely to return an item.")
    st.page_link("pages/CustomerPredictor.py", label="Open Customer Predictor", icon="📦")

with col2:
    st.subheader("💰 Campaign Profit")
    st.write("Forecast financial profit output for campaigns.")
    st.page_link("pages/ProfitPredictor.py", label="Open Profit Predictor", icon="💰")