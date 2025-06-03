import pandas as pd
import os

# Load models from FinalProject folder
linear_path = os.path.join("FinalProject", "linear_raw.csv")
quad_path = os.path.join("FinalProject", "quadratic_raw.csv")

linear = pd.read_csv(linear_path)
quad = pd.read_csv(quad_path)

# Extract feature lists
linear_feats = linear["Feature"].tolist()
linear_vals = linear["Coefficient"].tolist()

quad_feats = quad["Feature"].tolist()
quad_vals = quad["Coefficient"].tolist()

# Define expected inputs
input_features = ['HP', 'AC', 'Fortitude', 'Reflex', 'Will', 'Perception']

# User input
print()
print("\n📥 Enter your creature's stats:")
user_input = {}
for feat in input_features:
    val = float(input(f"  {feat}: "))
    user_input[feat] = val

# --- Linear Model Prediction ---
linear_pred = linear_vals[0]  # Intercept
for name, coef in zip(linear_feats[1:], linear_vals[1:]):
    linear_pred += coef * user_input[name]
linear_rounded = round(linear_pred)

# --- Quadratic Model Prediction ---
quad_pred = quad_vals[0]  # Intercept
for name, coef in zip(quad_feats[1:], quad_vals[1:]):
    if "^2" in name:
        base = name.replace("^2", "").strip()
        quad_pred += coef * (user_input[base] ** 2)
    elif " " in name:
        f1, f2 = name.split()
        quad_pred += coef * user_input[f1] * user_input[f2]
    else:
        quad_pred += coef * user_input[name]
quad_rounded = round(quad_pred)

# --- Output nicely ---
print("\n🧠 Predicted Creature Levels:")
print(f"  Linear Model       : {linear_pred:.2f} → Level {linear_rounded}")
print(f"  HP Quadratic Model : {quad_pred:.2f} → Level {quad_rounded}")
