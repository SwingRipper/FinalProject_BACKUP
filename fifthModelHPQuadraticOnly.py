
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.patches as mpatches

# Load data
input_path = os.path.join('FinalProject', 'clean_creatures.csv')
df = pd.read_csv(input_path)

# Drop name field
df = df.drop(columns=["Name"])

# Separate features and target
X = df.drop(columns=["Level"])
y = df["Level"]

# Separate out HP for quadratic treatment
hp = X[['HP']].values
other_features = X.drop(columns=['HP']).values

# Quadratic transform only on HP
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
hp_poly = poly.fit_transform(hp)  # [HP, HP^2]
hp_feature_names = poly.get_feature_names_out(['HP']).tolist()

# Combine quadratic HP with raw other features
X_poly = np.hstack([hp_poly, other_features])
feature_names = hp_feature_names + X.drop(columns=['HP']).columns.tolist()

# Normalize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_all_pred = model.predict(X_scaled)

# Metrics printing function
def print_metrics(label, y_true, y_pred):
    print(f"{label} Set:")
    print(f"  R-squared: {r2_score(y_true, y_pred):.4f}")
    print(f"  MSE      : {mean_squared_error(y_true, y_pred):.4f}")
    print()

# Print metrics
print_metrics("Training", y_train, y_train_pred)
print_metrics("Testing", y_test, y_test_pred)
print_metrics("Overall", y, y_all_pred)

# Intercept
print(f"\nIntercept (model base level): {model.intercept_:.4f}\n")

# Coefficients
print("Model Coefficients (normalized):")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name:>20}: {coef:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual Level")
plt.ylabel("Predicted Level")
plt.title("Actual vs Predicted Creature Level (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot all feature importances (sorted)
coefs = model.coef_
coef_abs = np.abs(coefs)
sorted_idx = np.argsort(coef_abs)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = coef_abs[sorted_idx]
sorted_signs = [coefs[i] for i in sorted_idx]
colors = ['blue' if val > 0 else 'red' for val in sorted_signs]

plt.figure(figsize=(10, len(sorted_features) * 0.3 + 2))
plt.barh(sorted_features, sorted_importances, color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Normalized Coefficient Magnitude")
plt.title("Feature Importance (Pure Linear Model)")
plt.tight_layout()
plt.show()

# Recover original (non-normalized) coefficients
feature_means = scaler.mean_
feature_stds = scaler.scale_
real_coefs = model.coef_ / feature_stds
real_intercept = model.intercept_ - np.sum((model.coef_ * feature_means) / feature_stds)

print("\n🧾 Non-Normalized Coefficients (for raw input scale):")
for name, coef in zip(feature_names, real_coefs):
    print(f"{name:>20}: {coef:.6f}")
print(f"{'Intercept':>20}: {real_intercept:.6f}")

# Load names again for residuals
df_names = pd.read_csv(input_path)[["Name", "Level"]]
df_names["Predicted_Level"] = y_all_pred
df_names["Residual"] = df_names["Predicted_Level"] - df_names["Level"]
df_names["Rounded_Predicted_Level"] = df_names["Predicted_Level"].round().astype(int)
df_names["Correct"] = df_names["Rounded_Predicted_Level"] == df_names["Level"]
accuracy = df_names["Correct"].mean() * 100

# Top discrepancies
overpowered = df_names.sort_values(by="Residual", ascending=False).head(5)
underpowered = df_names.sort_values(by="Residual", ascending=True).head(5)

print("\n🟥 Overpowered Creatures (Model thinks they're stronger than they are):")
print(overpowered[["Name", "Level", "Predicted_Level", "Residual"]])

print("\n🟦 Underpowered Creatures (Model thinks they're weaker than they are):")
print(underpowered[["Name", "Level", "Predicted_Level", "Residual"]])

print(f"\n✅ Rounded Prediction Accuracy:")
print(f"  {accuracy:.2f}% of creatures were correctly categorized by level.")
print(f"  ({df_names['Correct'].sum()} out of {len(df_names)} matched the true level)")

# Residual histogram
plt.figure(figsize=(10, 5))
plt.hist(df_names["Residual"], bins=30, edgecolor='black', color='skyblue')
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
plt.xlabel("Residual (Predicted Level - Actual Level)")
plt.ylabel("Number of Creatures")
plt.title("Distribution of Prediction Residuals")
plt.legend()
plt.tight_layout()
plt.show()
