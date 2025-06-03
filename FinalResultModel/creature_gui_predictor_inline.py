
import tkinter as tk

# Hardcoded model coefficients (raw input version)
linear_model = {
    "Intercept": -3.594724,
    "HP": 0.008894,
    "AC": 0.110255,
    "Fortitude": 0.172985,
    "Reflex": 0.094964,
    "Will": 0.225149,
    "Perception": 0.174567
}

quadratic_model = {
    "Intercept": -2.529640,
    "HP": 0.021343,
    "HP^2": 0.000547,
    "AC": -0.046760,
    "Fortitude": 0.230078,
    "Reflex": 0.021850,
    "Will": 0.359847,
    "Perception": -0.171243,
    "HP AC": 0.001180,
    "HP Fortitude": -0.002300,
    "HP Reflex": -0.005670,
    "HP Will": 0.001980,
    "HP Perception": -0.002150,
    "AC^2": 0.009800,
    "AC Fortitude": -0.004900,
    "AC Reflex": 0.007200,
    "AC Will": -0.018200,
    "AC Perception": 0.021500,
    "Fortitude^2": 0.014600,
    "Fortitude Reflex": -0.007100,
    "Fortitude Will": -0.009100,
    "Fortitude Perception": -0.000400,
    "Reflex^2": 0.011100,
    "Reflex Will": 0.014200,
    "Reflex Perception": -0.012100,
    "Will^2": 0.016300,
    "Will Perception": -0.011800,
    "Perception^2": 0.000400
}

def predict_levels():
    try:
        stats = {stat: float(entries[stat].get()) for stat in base_stats}

        linear_pred = linear_model["Intercept"]
        for key in base_stats:
            linear_pred += stats[key] * linear_model.get(key, 0)
        linear_rounded = round(linear_pred)

        quad_pred = quadratic_model["Intercept"]
        for feat, coef in quadratic_model.items():
            if feat == "Intercept":
                continue
            elif "^2" in feat:
                base = feat.replace("^2", "").strip()
                quad_pred += coef * (stats[base] ** 2)
            elif " " in feat:
                f1, f2 = feat.split()
                quad_pred += coef * stats[f1] * stats[f2]
            else:
                quad_pred += coef * stats[feat]
        quad_rounded = round(quad_pred)

        result = (
            f"Linear Model: {linear_pred:.2f} → Level {linear_rounded}\n"
            f"Quadratic Model: {quad_pred:.2f} → Level {quad_rounded}"
        )
        result_label.config(text=result)
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# Setup GUI
base_stats = ['HP', 'AC', 'Fortitude', 'Reflex', 'Will', 'Perception']

root = tk.Tk()
root.title("Creature Level Predictor")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

entries = {}
for stat in base_stats:
    tk.Label(frame, text=stat).pack()
    entry = tk.Entry(frame)
    entry.pack()
    entries[stat] = entry

tk.Button(frame, text="Predict Levels", command=predict_levels).pack(pady=10)

result_label = tk.Label(frame, text="", justify="left", font=("Arial", 10), fg="blue")
result_label.pack(pady=10)

root.mainloop()
