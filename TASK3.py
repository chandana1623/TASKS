import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
print(df.info())
# Features and target
X = df.drop("MedHouseVal", axis=1)   # predictors
y = df["MedHouseVal"]                # target (house price)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
# Use only one feature for visualization
X_simple = df[['MedInc']]
y_simple = df['MedHouseVal']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

model_s = LinearRegression()
model_s.fit(X_train_s, y_train_s)
y_pred_s = model_s.predict(X_test_s)

# Plot regression line
plt.scatter(X_test_s, y_test_s, color="blue", alpha=0.5, label="Actual")
plt.plot(X_test_s, y_pred_s, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.title("Simple Linear Regression (Income vs House Value)")
plt.legend()
plt.show()
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Mapping features to coefficients
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coef_df)
