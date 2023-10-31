import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('kc_house_data.csv')  # Ganti dengan path ke file dataset yang sesuai

# Select 8 features for prediction
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'yr_built']

X = data[features]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Create a Streamlit web app
st.title('King County House Price Prediction')

# Add input fields for user to enter data
st.header('Enter House Details:')
bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10)
bathrooms = st.number_input('Bathrooms', min_value=1, max_value=10)
sqft_living = st.number_input('Sqft Living', min_value=200, max_value=10000)
sqft_lot = st.number_input('Sqft Lot', min_value=200, max_value=10000)
floors = st.number_input('Floors', min_value=1, max_value=5)
condition = st.number_input('Condition', min_value=1, max_value=5)
grade = st.number_input('Grade', min_value=1, max_value=13)
yr_built = st.number_input('Year Built', min_value=1900, max_value=2023)

# Make predictions based on user inputs
input_data = [[bedrooms, bathrooms, sqft_living, sqft_lot, floors, condition, grade, yr_built]]
predicted_price = model.predict(input_data)

# Display the predicted price
if st.button('Predict Price'):
    st.subheader('Predicted House Price:')
    st.write(f'${predicted_price[0]:,.2f}')

# Optionally, you can add more features and improve the user interface of your app