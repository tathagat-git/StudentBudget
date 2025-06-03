import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Load the dataset
df = pd.read_csv("student_monthly_expense_data_700.csv")

# Define features (X) and target (y)
X = df.drop("monthly_expense", axis=1)
y = df["monthly_expense"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("📈 R² Score:", r2_score(y_test, y_pred))
print("📉 MAE:", mean_absolute_error(y_test, y_pred))
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("🔁 RMSE:", rmse)

# Optional: Show model coefficients
print("\n📊 Feature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
