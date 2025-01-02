import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

st.write(':satellite_antenna: Telecom PVT.LTD.')

st.markdown('#### Data Privacy')
st.caption("""
You are entering a restricted environment containing highly sensitive and confidential information owned by Telecom PVT LTD. Access is strictly limited to authorized personnel only. Unauthorized access or disclosure of this information is a violation and may result in legal action under local regulations. Please ensure you are properly authorized to proceed.
""")


left_column, right_column = st.columns(2)
pressed = left_column.button('Confirm')
if pressed:
  #checkbox
        agree = st.checkbox('I fully understand my authority and my limitations, as well as the companys terms and local laws.')
        if agree:
            st.write('Great!')

'''
# Customer Churn Identification

This page is designed to provide insights into customer churn specifically for Telecom PVT. LTD. Our dashboard aims to identify customers at risk of churning based on various factors relevant to our operations.

For those identified as likely to churn, targeted retention strategies will be implemented, such as special promotions to encourage their return. Conversely, for customers not likely to churn, our focus will be on enhancing their overall experience through tailored support and optimal service delivery.

Additionally, users can add new customer information to the dashboard, allowing us to predict whether these customers may churn in the future. By leveraging data-driven insights, Telecom PVT. LTD. seeks to improve customer retention, satisfaction, and loyalty.
'''
#____________________________________________________________________________________________________________

df = pd.read_csv("df_eda.csv")
#____________________________________________________________________________________________________________
# Count churn values
churn_count = df['churn'].value_counts().sort_index()

# Define a color palette
colors = ["#008080", "#003366"]  # Using hex codes for better color options

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

# Bar plot
bars = axs[0].bar(churn_count.index.astype(str), churn_count.values, color=colors)
axs[0].set_title('Customer Churn Analysis', fontsize=16, fontweight='bold')
axs[0].set_xlabel('Churn Status', fontsize=14)
axs[0].set_ylabel('Number of Customers', fontsize=14)

# Add amounts on top of each bar with matching color
for bar, color in zip(bars, colors):
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval, str(int(yval)), 
                 ha='center', va='bottom', fontsize=12, color=color)

# Pie chart
axs[1].pie(churn_count, labels=churn_count.index.astype(str), autopct='%1.1f%%', 
           startangle=140, colors=colors)
axs[1].set_title('Churn Distribution', fontsize=16, fontweight='bold')
axs[1].axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.

# Display the plots in Streamlit
st.header('2020 - 2023 Customer Churn')
st.pyplot(fig)  # Use Streamlit's function to display the Matplotlib figure
# ____________________________________________________________________________________________________________
# Function to plot Salary Box Plot (churn vs non-churn)
def plot_salary_boxplot(df_eda):
    plt.figure(figsize=(10, 5))
    
    # Box plot for salary distribution by churn status
    sns.boxplot(x='churn', y='estimated_salary', data=df_eda, palette={"0": "blue", "1": "black"})
    
    plt.title("Salary Box Plot (Churn vs Non-Churn)", fontsize=16)
    plt.xlabel("Churn Status", fontsize=14)
    plt.ylabel("Estimated Salary", fontsize=14)

    # Display the plot in Streamlit
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying
#____________________________________________________________________________________________________________
# Load the model
model_path = "model.pkl" 
with open(model_path, 'rb') as file:
    model = pickle.load(file)

def run_modelling(user_input):
    prediction = model.predict(user_input)
    return prediction

# Function to get user input from the sidebar
def get_user_input(df_eda):
    st.sidebar.header(':satellite_antenna: Telecom PVT.LTD.')
    st.sidebar.header("Input Customer Information")
    
    # Dynamically retrieve unique values from the DataFrame for each selectbox
    customer_id = st.sidebar.selectbox("Customer ID", df_eda['customer_id'].unique())
    telecom_partner = st.sidebar.selectbox("Telecom Partner", df_eda['telecom_partner'].unique())
    gender = st.sidebar.selectbox("Gender", df_eda['gender'].unique())
    age = st.sidebar.number_input("Age", min_value=int(df_eda['age'].min()), max_value=int(df_eda['age'].max()), value=int(df_eda['age'].mean()))
    state = st.sidebar.selectbox("State", df_eda['state'].unique())
    city = st.sidebar.selectbox("City", df_eda['city'].unique())
    pincode = st.sidebar.selectbox("Pincode", df_eda['pincode'].unique())
    date_of_registration = st.sidebar.date_input("Date of Registration")
    num_dependents = st.sidebar.number_input("Number of Dependents", min_value=int(df_eda['num_dependents'].min()), max_value=int(df_eda['num_dependents'].max()), value=int(df_eda['num_dependents'].mean()))
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=int(df_eda['estimated_salary'].min()), max_value=int(df_eda['estimated_salary'].max()), value=int(df_eda['estimated_salary'].mean()))
    calls_made = st.sidebar.number_input("Calls Made", min_value=int(df_eda['calls_made'].min()), max_value=int(df_eda['calls_made'].max()), value=int(df_eda['calls_made'].mean()))
    sms_sent = st.sidebar.number_input("SMS Sent", min_value=int(df_eda['sms_sent'].min()), max_value=int(df_eda['sms_sent'].max()), value=int(df_eda['sms_sent'].mean()))
    data_used = st.sidebar.number_input("Data Used (in MB)", min_value=int(df_eda['data_used'].min()), max_value=int(df_eda['data_used'].max()), value=int(df_eda['data_used'].mean()))

    # Create a DataFrame from the inputs
    user_input = pd.DataFrame({
        'customer_id': [customer_id], 
        'telecom_partner': [telecom_partner],
        'gender': [gender],
        'age': [age],
        'state': [state],
        'city': [city],
        'pincode': [pincode],
        'date_of_registration': [date_of_registration],  
        'num_dependents': [num_dependents],
        'estimated_salary': [estimated_salary],
        'calls_made': [calls_made],
        'sms_sent': [sms_sent],
        'data_used': [data_used]
    })

    return user_input

# Main function to run the Streamlit app
def main():
    st.title("Customer Churn Prediction App")
    
    # Load your DataFrame here (assuming df_eda is already loaded)
    df_eda = pd.read_csv("df_eda.csv")  # Replace with your file path
    
    # Get user input
    user_input = get_user_input(df_eda)

    # Display the user input on the page
    st.write("### Customer Input Information")
    st.dataframe(user_input)

    # Run prediction when the button is clicked
    if st.button("Predict Churn"):
        prediction = run_modelling(user_input)
        if prediction[0] == 1:
            st.write("### Prediction: The customer is likely to churn.")
        else:
            st.write("### Prediction: The customer is not likely to churn.")

if __name__ == "__main__":
    main()
#____________________________________________________________________________



        

