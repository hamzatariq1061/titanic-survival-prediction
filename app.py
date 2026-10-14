
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

package = load_model()

model = package["model"]
encoders = package["encoders"]
features = package["features"]

st.title("🚢 Titanic Survival Prediction System")

st.write(
    "Enter passenger information to predict "
    "whether the passenger survived."
)

pclass = st.selectbox(
    "Passenger Class",
    ["First", "Second", "Third"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

sibling = st.selectbox(
    "Sibling",
    ["Zero", "One", "Two", "Three"]
)

embarked = st.selectbox(
    "Embarked",
    ["Southampton", "Queenstown", "Cherbourg"]
)

if st.button("Predict Survival"):

    input_values = {
        "PClass": pclass,
        "Gender": gender,
        "Sibling": sibling,
        "Embarked": embarked
    }

    encoded_values = {}

    for column, value in input_values.items():
        encoded_values[column] = (
            encoders[column].transform([value])[0]
        )

    input_df = pd.DataFrame(
        [encoded_values],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("Passenger Survived!")
    else:
        st.error("Passenger Did Not Survive!")
