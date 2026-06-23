import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Feature Importance", page_icon="🌟")

st.title("🌟 Key Drivers of Churn")
st.markdown("Which factors contribute most to a customer leaving?")

data = {
    'Feature': ['Contract_Month-to-month', 'tenure', 'TotalCharges', 'MonthlyCharges', 'InternetService_Fiber optic'],
    'Importance (%)': [19.5, 15.2, 11.4, 9.8, 6.1]
}
df_imp = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(df_imp['Feature'], df_imp['Importance (%)'], color='coral')
ax.set_xlabel('Importance Contribution (%)')
ax.invert_yaxis() 
st.pyplot(fig)

st.markdown("""
### Business Recommendations:
1. **Incentivize Long-Term Contracts:** The month-to-month contract is the biggest driver of churn. Offer heavy discounts for the first 3 months if users lock into a 1-year contract.
2. **Early Tenure Support:** Customers are highly likely to leave in their first few months (Tenure). Implement a strong 90-day onboarding and check-in program.
3. **Fiber Optic Review:** Fiber optic customers churn significantly more than DSL. Technical teams need to investigate outages or pricing structures in this tier.
""")