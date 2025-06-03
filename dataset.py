import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of synthetic student records
n_students = 700

# Generate synthetic features
online_orders = np.random.poisson(5, n_students)
cafeteria_visits = np.random.randint(5, 25, n_students)
commute_km = np.random.normal(30, 10, n_students).clip(0)
parties_attended = np.random.poisson(2, n_students)
subscriptions = np.random.randint(0, 5, n_students)

# Generate synthetic expenses with a weighted sum + noise
monthly_expense = (
    online_orders * 200 +
    cafeteria_visits * 50 +
    commute_km * 3 +
    parties_attended * 300 +
    subscriptions * 150 +
    np.random.normal(0, 200, n_students)
).round(2)

# Create DataFrame
df = pd.DataFrame({
    "online_orders": online_orders,
    "cafeteria_visits": cafeteria_visits,
    "commute_km": commute_km,
    "parties_attended": parties_attended,
    "subscriptions": subscriptions,
    "monthly_expense": monthly_expense
})

# Save to CSV
csv_path = "/mnt/data/student_monthly_expense_data_700.csv"
df.to_csv(csv_path, index=False)

csv_path
