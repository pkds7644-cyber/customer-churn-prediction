import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Dataset Insights", page_icon="📊")

@st.cache_data 
def load_data():
    if os.path.exists('data/churn_data.csv'):
        return pd.read_csv('data/churn_data.csv')
    
    elif os.path.exists('../data/churn_data.csv'):
        return pd.read_csv('../data/churn_data.csv')
        
    else:
        return pd.DataFrame()

st.title("📊 Dataset Insights")
st.markdown("Explore the historical demographics and service usage of our customer base.")

df = load_data()

if df.empty:
    st.warning("⚠️ Dataset not found. Please ensure 'churn_data.csv' is in the 'data' folder.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Male / Female", f"{len(df[df['gender']=='Male'])} / {len(df[df['gender']=='Female'])}")
    col3.metric("Avg Monthly Revenue", f"${df['MonthlyCharges'].mean():.2f}")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Churn by Contract Type")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        df.groupby(['Contract', 'Churn']).size().unstack().plot(kind='bar', ax=ax1, stacked=True)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
    with col_chart2:
        st.subheader("Tenure Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(df['tenure'], bins=30, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Months with Company")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)