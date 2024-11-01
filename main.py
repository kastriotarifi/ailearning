import pandas as pd
import autokeras as ak
import streamlit as st
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('chatbot_data.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare training data
train_x = train_data['question'].values
train_y = train_data['response'].values

# Create and train the model
model = ak.TextClassifier(max_trials=10)  # You can adjust max_trials for more tuning
model.fit(train_x, train_y, epochs=10)

# Streamlit app layout
st.title("Chatbot using AutoKeras")

# User input
user_input = st.text_input("You: ", "")

# Generate response
if st.button("Send"):
    if user_input:
        response = model.predict([user_input])
        st.text(f"Chatbot: {response[0][0]}")  # Display response
    else:
        st.text("Please enter a message.")
