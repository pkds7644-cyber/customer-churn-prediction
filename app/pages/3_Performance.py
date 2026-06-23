import streamlit as st

st.set_page_config(page_title="Model Performance", page_icon="📈")

st.title("📈 Model Evaluation Metrics")
st.markdown("Technical breakdown of the Random Forest Pipeline performance on unseen test data.")

col1, col2 = st.columns(2)

with col1:
    st.info("**Recall: 0.52**")
    st.write("We correctly identify 52% of all actual churners. In business, we optimize for this metric to ensure we don't miss at-risk customers.")
    
    st.info("**Precision: 0.65**")
    st.write("When the model predicts a customer will churn, it is correct 65% of the time. This prevents us from wasting too much money on false-positive retention discounts.")

with col2:
    st.info("**ROC-AUC: 0.84**")
    st.write("An excellent score indicating the model has a strong ability to distinguish between customers who will stay and those who will leave.")
    
    st.info("**F1 Score: 0.58**")
    st.write("The harmonic mean between Precision and Recall, providing a balanced view of model performance on our imbalanced dataset.")