import streamlit as st

st.set_page_config(
    page_title="ChurnGuard | ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🛡️ ChurnGuard: Customer Retention System")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Customer Churn Prediction Dashboard.
    
    **The Business Problem:**
    Acquiring a new customer is up to 5 times more expensive than retaining an existing one. 
    This application utilizes Machine Learning to identify customers who are at a high risk 
    of terminating their contracts, allowing retention teams to take proactive measures.
    
    **How to use this tool:**
    👈 **Navigate using the sidebar to explore:**
    * **📊 Insights:** View historical data and customer demographics.
    * **🔮 Prediction:** Enter a specific customer's details to get a real-time churn risk assessment.
    * **📈 Performance:** Review the technical evaluation metrics of the underlying Random Forest model.
    * **🌟 Features:** Discover which business factors are driving customer churn.
    """)
    
    st.markdown("### 📊 Project Impact Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Customer Records", value="7,043")
    with col2:
        st.metric(label="Overall Churn Rate", value="26.5%")
    with col3:
        st.metric(label="Model Accuracy", value="~80.0%")

if __name__ == "__main__":
    main()