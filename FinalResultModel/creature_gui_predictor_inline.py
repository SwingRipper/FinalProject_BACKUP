
import tkinter as tk

# Hardcoded model coefficients (raw input version)
# Linear model (latest correct values)
linear_model = {
    "Intercept": -3.594723604325986,
    "HP": 0.008894060255536409,
    "AC": 0.11025542728921836,
    "Fortitude": 0.17298540370082183,
    "Reflex": 0.09496421293894965,
    "Will": 0.1682189044913387,
    "Perception": 0.01737578980516953
}

# Quadratic model (latest correct values)
quadratic_model = {
    "Intercept": -2.529640,
    "HP": 0.021343,
    "HP^2": 0.000036,
    "AC": -0.046760,
    "Fortitude": 0.230078,
    "Reflex": 0.021850,
    "Will": 0.349293,
    "Perception": -0.189674,
    "HP AC": 0.000012,
    "HP Fortitude": -0.000142,
    "HP Reflex": -0.000911,
    "HP Will": 0.000119,
    "HP Perception": -0.000398,
    "AC^2": 0.003689,
    "AC Fortitude": -0.003027,
    "AC Reflex": 0.004589,
    "AC Will": -0.019470,
    "AC Perception": 0.021797,
    "Fortitude^2": 0.005269,
    "Fortitude Reflex": -0.005746,
    "Fortitude Will": -0.006617,
    "Fortitude Perception": -0.000170,
    "Reflex^2": 0.004609,
    "Reflex Will": 0.010664,
    "Reflex Perception": -0.009588,
    "Will^2": 0.011142,
    "Will Perception": -0.009350,
    "Perception^2": 0.000170
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
