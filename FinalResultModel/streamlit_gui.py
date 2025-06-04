
import streamlit as st

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

# Footer blurb
st.markdown("---")
st.markdown(
    "**About This Tool (v2)**\n\n"
    "This tool (by SwingRipper) uses two regression models trained on Pathfinder 2e creature data to help GMs evaluate how defensively strong a custom creature is.\n"
    "The linear model is simple and less likely to be thrown off by weird extreme monsters.\n"
    "The quadratic model is more accurate in normal cases, but can be thrown off by weird values.\n"
    "NOTE: mage like creatures are likely -1 level defensively and warrior like creatures tend to be +1 level defensively. \n"
    "Simply input all your creature numbers and press calculate!\n"
)
