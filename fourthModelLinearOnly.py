import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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

# Use raw linear features only
X_poly = X.values
feature_names = X.columns.tolist()

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
    
    
    
# Export non-normalized version
real_coefs = model.coef_ / scaler.scale_
real_intercept = model.intercept_ - np.sum((model.coef_ * scaler.mean_) / scaler.scale_)

export_raw_df = pd.DataFrame({
    "Feature": ["Intercept"] + feature_names,
    "Coefficient": [real_intercept] + list(real_coefs)
})

output_raw_path = os.path.join("FinalProject", "exported_model_coefficients_raw.csv")
export_raw_df.to_csv(output_raw_path, index=False)

print(f"\n📤 Raw (non-normalized) coefficients exported to: {output_raw_path}")    
    
    
    

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

# Get signed coefficients and sorted indices
coefs = model.coef_
coef_abs = np.abs(coefs)
sorted_idx = np.argsort(coef_abs)[::-1]

# Sort everything accordingly
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = coef_abs[sorted_idx]
sorted_signs = [coefs[i] for i in sorted_idx]

# Set colors: red = negative, blue = positive
colors = ['red' if val < 0 else 'blue' for val in sorted_signs]

# Plot
plt.figure(figsize=(10, len(sorted_features) * 0.3 + 2))
plt.barh(sorted_features, sorted_importances, color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Normalized Absolute Coefficient")
plt.title("Feature Importance (Signed by Correlation)")
plt.tight_layout()
plt.show()


# Group coefficients by type
linear_terms = []
squared_terms = []
interaction_terms = []

for i, name in enumerate(feature_names):
    coef = model.coef_[i]
    if re.fullmatch(r"^[A-Za-z]+$", name):
        linear_terms.append((name, coef))
    elif "^2" in name:
        squared_terms.append((name, coef))
    else:
        interaction_terms.append((name, coef))

# Sort by absolute value within each group
def sort_terms(terms):
    return sorted(terms, key=lambda x: abs(x[1]), reverse=True)

linear_terms = sort_terms(linear_terms)
squared_terms = sort_terms(squared_terms)
interaction_terms = sort_terms(interaction_terms)

# Combine
grouped_terms = linear_terms + squared_terms + interaction_terms
labels = [term[0] for term in grouped_terms]
values = [abs(term[1]) for term in grouped_terms]
colors = ['blue' if coef > 0 else 'red' for _, coef in grouped_terms]

# Plot
plt.figure(figsize=(10, len(labels) * 0.3 + 2))
plt.barh(labels, values, color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Normalized Absolute Coefficient")
plt.title("Grouped Feature Importance (Colored by Correlation)")

# Add section labels
group_boundaries = [0, len(linear_terms), len(linear_terms) + len(squared_terms), len(grouped_terms)]
group_names = ["Linear Terms", "Squared Terms", "Interaction Terms"]

for i, name in enumerate(group_names):
    midpoint = (group_boundaries[i] + group_boundaries[i+1] - 1) / 2
    plt.text(-0.05 * max(values), midpoint, name, ha='right', va='center', fontweight='bold')


# Legend
legend_handles = [
    mpatches.Patch(color='blue', label='Positive correlation'),
    mpatches.Patch(color='red', label='Negative correlation')
]
plt.legend(handles=legend_handles, loc='lower right')

plt.tight_layout()
plt.show()

# Recover original (non-normalized) coefficients
# Only valid if using StandardScaler and PolynomialFeatures

# 1. Get the original means and stds from the scaler
feature_means = scaler.mean_
feature_stds = scaler.scale_

# 2. Convert coefficients
real_coefs = model.coef_ / feature_stds
real_intercept = model.intercept_ - np.sum((model.coef_ * feature_means) / feature_stds)

# 3. Print results
print("\n🧾 Non-Normalized Coefficients (for raw input scale):")
for name, coef in zip(feature_names, real_coefs):
    print(f"{name:>20}: {coef:.6f}")
print(f"{'Intercept':>20}: {real_intercept:.6f}")


# Re-load creature names and levels
df_names = pd.read_csv(input_path)[["Name", "Level"]]
df_names["Predicted_Level"] = y_all_pred
df_names["Residual"] = df_names["Predicted_Level"] - df_names["Level"]

# Sort by residuals
overpowered = df_names.sort_values(by="Residual", ascending=False).head(5)
underpowered = df_names.sort_values(by="Residual", ascending=True).head(5)

# Show results
print("\n🟥 Overpowered Creatures (Model thinks they're stronger than they are):")
print(overpowered[["Name", "Level", "Predicted_Level", "Residual"]])

print("\n🟦 Underpowered Creatures (Model thinks they're weaker than they are):")
print(underpowered[["Name", "Level", "Predicted_Level", "Residual"]])

# Histogram of residuals
plt.figure(figsize=(10, 5))
plt.hist(df_names["Residual"], bins=30, edgecolor='black', color='skyblue')
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
plt.xlabel("Residual (Predicted Level - Actual Level)")
plt.ylabel("Number of Creatures")
plt.title("Distribution of Prediction Residuals")
plt.legend()
plt.tight_layout()
plt.show()

# Round predicted levels to nearest integer
df_names["Rounded_Predicted_Level"] = df_names["Predicted_Level"].round().astype(int)

# Compare to actual level
df_names["Correct"] = df_names["Rounded_Predicted_Level"] == df_names["Level"]

# Calculate accuracy
accuracy = df_names["Correct"].mean() * 100

# Print result
print(f"\n✅ Rounded Prediction Accuracy:")
print(f"  {accuracy:.2f}% of creatures were correctly categorized by level.")
print(f"  ({df_names['Correct'].sum()} out of {len(df_names)} matched the true level)")



