import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# File path setup
input_path = os.path.join('FinalProject', 'clean_creatures.csv')
df = pd.read_csv(input_path)

# Drop name field
df = df.drop(columns=["Name"])

# Separate features and target
X = df.drop(columns=["Level"])
y = df["Level"]

# Polynomial transform for HP
poly = PolynomialFeatures(degree=2, include_bias=False)
hp_poly = poly.fit_transform(X[["HP"]])  # HP and HP^2

# Concatenate with remaining features
X_other = X.drop(columns=["HP"]).values
X_combined = np.hstack([hp_poly, X_other])

# Feature names
feature_names = poly.get_feature_names_out(["HP"]).tolist() + X.drop(columns=["HP"]).columns.tolist()

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_all_pred = model.predict(X_scaled)

# Evaluation
def print_metrics(label, y_true, y_pred):
    print(f"{label} Set:")
    print(f"  R-squared: {r2_score(y_true, y_pred):.4f}")
    print(f"  MSE      : {mean_squared_error(y_true, y_pred):.4f}")
    print()

print_metrics("Training", y_train, y_train_pred)
print_metrics("Testing", y_test, y_test_pred)
print_metrics("Overall", y, y_all_pred)

# Intercept (base level)
print(f"\nIntercept (base level if all standardized features = 0): {model.intercept_:.4f}")

# Show coefficients
print("\nModel Coefficients (on normalized scale):")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name:>15}: {coef:.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual Level")
plt.ylabel("Predicted Level")
plt.title("Actual vs Predicted Creature Level (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Normalized Feature Importances
coef_abs = np.abs(model.coef_)
sorted_idx = np.argsort(coef_abs)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = coef_abs[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances)
plt.gca().invert_yaxis()
plt.xlabel("Normalized Absolute Coefficient")
plt.title("Normalized Feature Importance (Standardized Features)")
plt.tight_layout()
plt.show()
