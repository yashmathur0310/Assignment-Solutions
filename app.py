import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('data-final.csv')

# Extract unique source countries and speeds from the dataset
source_countries = [col.replace('source_name_', '') for col in data.columns if col.startswith('source_name_')]
speeds = [col.replace('speed actual_', '') for col in data.columns if col.startswith('speed actual_')]

# Load the trained model
x=data.drop(['firm'],axis=1).values
y=data['firm'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
model=DecisionTreeClassifier()
model.fit(x,y)

# Streamlit app
st.title('Money Transfer Firm Predictor')
st.write('Send Money to India')

# User inputs
amount = st.number_input('Amount to Transfer', min_value=0)
fee = st.number_input('Exchange Rate', min_value=0)
selected_source_country = st.selectbox('Source Country', source_countries)
selected_speed = st.selectbox('Speed of Transfer', speeds)

# Initialize input data with the numerical features
input_data = pd.DataFrame({
    'Amount to Transfer': [amount],
    'Exchange Rate': [fee]
})

# Create one-hot encoded columns for source_country and speed
for country in source_countries:
    input_data[f'source_name_{country}'] = [1 if country == selected_source_country else 0]

for speed in speeds:
    input_data[f'speed actual_{speed}'] = [1 if speed == selected_speed else 0]

# Align the input data with the model's expected columns
model_columns = data.columns.drop('firm')  # Assuming 'firm' is the target variable
input_data_aligned = input_data.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button('Predict Firm'):
    # Make prediction using the model
    st.write(input_data_aligned)
    prediction = model.predict(input_data_aligned)
    
    # Display the prediction
    st.write(f"The predicted firm for this transaction is: **{prediction[0]}**")
