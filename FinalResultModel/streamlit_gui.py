import streamlit as st

# Define your models
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

st.title("Creature Level Predictor")

# Input fields
inputs = {}
for stat in ["HP", "AC", "Fortitude", "Reflex", "Will", "Perception"]:
    inputs[stat] = st.number_input(stat, value=10, step=1, format="%d")


if st.button("Predict"):
    # Linear prediction
    linear = linear_model["Intercept"] + sum(inputs[k] * linear_model[k] for k in inputs)
    linear_rounded = round(linear)

    # Quadratic prediction
    quad = quadratic_model["Intercept"]
    for feat, coef in quadratic_model.items():
        if feat == "Intercept":
            continue
        elif "^2" in feat:
            base = feat.replace("^2", "")
            quad += coef * (inputs[base] ** 2)
        elif " " in feat:
            f1, f2 = feat.split()
            quad += coef * inputs[f1] * inputs[f2]
        else:
            quad += coef * inputs[feat]
    quad_rounded = round(quad)

    st.success(f"Linear Model: {linear:.2f} → Level {linear_rounded}")
    st.success(f"Quadratic Model: {quad:.2f} → Level {quad_rounded}")
