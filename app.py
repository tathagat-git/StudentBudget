import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset and train model (for demo simplicity, retrain on every run)
@st.cache_data
def load_data_and_train():
    df = pd.read_csv("student_monthly_expense_data_700.csv")
    X = df.drop("monthly_expense", axis=1)
    y = df["monthly_expense"]
    model = LinearRegression()
    model.fit(X, y)
    return model

model = load_data_and_train()

st.title("🎓 Student Monthly Expense Estimator")

online_orders = st.number_input("Number of online orders (per month)", min_value=0, max_value=50, value=5)
cafeteria_visits = st.number_input("Cafeteria visits per week", min_value=0, max_value=30, value=10)
commute_km = st.number_input("Monthly commute distance (km)", min_value=0.0, max_value=200.0, value=30.0)
parties_attended = st.number_input("Number of parties attended (per month)", min_value=0, max_value=20, value=2)
subscriptions = st.number_input("Number of subscriptions", min_value=0, max_value=10, value=1)

if st.button("Estimate Monthly Expense"):
    input_data = np.array([[online_orders, cafeteria_visits, commute_km, parties_attended, subscriptions]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Monthly Expense: ₹{prediction:.2f}")
