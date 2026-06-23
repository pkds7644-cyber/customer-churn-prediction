
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Churn Prediction", page_icon="🔮")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'churn_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

st.title("🔮 Customer Churn Prediction")
st.markdown("Enter customer details below to calculate their probability of leaving.")

with st.form("prediction_form"):
    st.subheader("Demographics & Account Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], help="1 = Yes, 0 = No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
        
    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    st.subheader("Add-on Services")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        
    with col5:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
    with col6:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
    submit_button = st.form_submit_button("Predict Churn Risk")

if submit_button:
    if model is None:
        st.error("🚨 Model not found! Please run train.py first to generate the model.")
    else:
        rf_model = model["model"]
        scaler = model["scaler"]
        encoder = model["encoder"]

        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless],
            'PaymentMethod': [payment],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        input_data['ChargesRatio'] = input_data['TotalCharges'] / (input_data['MonthlyCharges'] + 1)

        num_cols = input_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = input_data.select_dtypes(include=['object']).columns.tolist()

        input_encoded = encoder.transform(input_data[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        input_cat_df = pd.DataFrame(input_encoded, columns=encoded_cols, index=input_data.index)

        input_scaled = scaler.transform(input_data[num_cols])
        input_num_df = pd.DataFrame(input_scaled, columns=num_cols, index=input_data.index)

        final_input = pd.concat([input_num_df, input_cat_df], axis=1)
        
        probability = rf_model.predict_proba(final_input)[0][1] * 100
        prediction = rf_model.predict(final_input)[0]
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Churn Probability", f"{probability:.1f}%")
            
        with res_col2:
            if probability > 60:
                st.error("🚨 HIGH RISK: Immediate retention action required.")
            elif probability > 30:
                st.warning("⚠️ MODERATE RISK: Monitor account activity.")
            else:
                st.success("✅ LOW RISK: Customer is likely to stay.")