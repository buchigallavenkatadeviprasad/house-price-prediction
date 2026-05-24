# house_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
# (Using sklearn California housing dataset since Boston is deprecated)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Explore data
# -----------------------------
print("Checking for null values:\n", df.isnull().sum())

# Histogram (optional visualization)
df.hist(bins=30, figsize=(10, 8))
plt.tight_layout()
plt.savefig("house_histograms.png")
plt.close()

# -----------------------------
# 3. Features and labels
# -----------------------------
X = df.drop("MedHouseVal", axis=1)  # features
y = df["MedHouseVal"]  # target (Median House Value)

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Model Evaluation:")
print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# -----------------------------
# 7. Save model
# -----------------------------
joblib.dump(model, "house_model.pkl")
print("✅ Model saved as house_model.pkl")
