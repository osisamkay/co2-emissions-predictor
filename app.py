import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App Title
st.title("CO₂ Emissions Prediction and Analysis")

# Load the dataset
data_path = "world_bank_data.csv"  # Replace with your dataset path
model_path = "co2_emission_model.pkl"  # Replace with your model path

# Load data
try:
    data = pd.read_csv(data_path)
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error(f"Could not find the dataset at {data_path}. Please check the file path.")
    st.stop()

# Add calculated features
data["Emissions Intensity"] = data["CO2 Emissions (metric tons per capita)"] / data["GDP (current US$)"]
data["Renewable Energy Urban Impact"] = (
    data["Renewable Energy (% of total final energy consumption)"] *
    data["Urban Population (% of total)"] / 100
)
data.rename(columns={"country": "Country", "date": "Year"}, inplace=True)

# Load the pre-trained model
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Could not find the model file at {model_path}. Please check the file path.")
    st.stop()

# Prepare the data
ml_data = data.dropna(subset=["CO2 Emissions (metric tons per capita)", "GDP (current US$)"])
ml_data["Log GDP"] = np.log1p(ml_data["GDP (current US$)"])
X = ml_data.drop(columns=["CO2 Emissions (metric tons per capita)", "Country", "Year", "GDP (current US$)"])
y = ml_data["CO2 Emissions (metric tons per capita)"]

# Predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Feature importances
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Sidebar for user input
st.sidebar.header("Predict CO₂ Emissions")
st.sidebar.markdown("Enter values for the features below to predict CO₂ emissions:")
user_inputs = {}
for feature in X.columns:
    default_value = float(X[feature].mean())
    user_inputs[feature] = st.sidebar.number_input(f"{feature}", value=default_value)

# Convert user inputs into a DataFrame
user_input_df = pd.DataFrame([user_inputs])

# Predict emissions based on user inputs
user_prediction = model.predict(user_input_df)[0]
st.sidebar.write(f"Predicted CO₂ Emissions: **{user_prediction:.2f} metric tons per capita**")

# Display Results
st.header("Model Performance")
st.write(f"Mean Squared Error: **{mse:.2f}**")
st.write(f"R² Score: **{r2:.2f}**")

# Display feature importances
st.header("Feature Importances")
st.write("The importance of each feature in the Random Forest model:")
st.dataframe(feature_importance)

# Bar Chart of Feature Importances
st.bar_chart(feature_importance.set_index("Feature"))

# Scatter Plot: Actual vs. Predicted
st.header("Actual vs. Predicted CO₂ Emissions")
fig, ax = plt.subplots()
ax.scatter(y, y_pred, alpha=0.6, edgecolors="k")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
ax.set_xlabel("Actual CO₂ Emissions")
ax.set_ylabel("Predicted CO₂ Emissions")
ax.set_title("Actual vs. Predicted CO₂ Emissions")
st.pyplot(fig)

# Histogram of Selected Feature
st.header("Feature Distribution")
selected_feature = st.selectbox("Select a feature to visualize its distribution", X.columns)
fig, ax = plt.subplots()
ax.hist(X[selected_feature], bins=20, color="skyblue", edgecolor="black")
ax.set_title(f"Distribution of {selected_feature}")
ax.set_xlabel(selected_feature)
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Insights Section
st.header("Key Insights")
st.markdown(
    """
    - **High Importance Features**: Focus on features with the highest importance to reduce CO₂ emissions.
    - **Actual vs. Predicted Plot**: Observe how closely the model predicts real values.
    - **Feature Distribution**: Understand the data distribution of specific features.
    """
)


