import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("kc_house_data.csv")

# Define the features and target variable
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'condition']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("KC House Price Predictor")

st.sidebar.header("User Input Features")

sqft_living = st.sidebar.slider("Sqft Living Area", float(X['sqft_living'].min()), float(X['sqft_living'].max()), float(X['sqft_living'].mean()))
bedrooms = st.sidebar.slider("Bedrooms", int(X['bedrooms'].min()), int(X['bedrooms'].max()), int(X['bedrooms'].mean()))
bathrooms = st.sidebar.slider("Bathrooms", float(X['bathrooms'].min()), float(X['bathrooms'].max()), float(X['bathrooms'].mean()))
floors = st.sidebar.slider("Floors", float(X['floors'].min()), float(X['floors'].max()), float(X['floors'].mean()))
condition = st.sidebar.slider("Condition", int(X['condition'].min()), int(X['condition'].max()), int(X['condition'].mean()))

def user_input_features():
    data = {
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "condition": condition
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)

st.subheader("Prediction")
st.write("Predicted Price:", prediction[0])