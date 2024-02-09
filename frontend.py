import streamlit as st
import requests

# Streamlit web application
st.title('Investment Insights and Sentiment Analysis')

# Input field for AI prompts
prompt = st.text_area('Enter prompt for investment insight:', '')

# Button to generate insights and perform sentiment analysis
if st.button('Generate Insights and Analyze Sentiment'):
    # API endpoint for generating insights and sentiment analysis
    endpoint = 'sandipkbtcoe.py'
    
    # Send POST request to API
    response = requests.post(endpoint, json={'prompt': prompt})
    
    # Parse JSON response
    if response.status_code == 200:
        data = response.json()
        insights = data['insights']
        sentiment = data['sentiment']
        st.write(f'Generated Insights: {insights}')
        st.write(f'Sentiment: {sentiment}')
    else:
        st.write('Error: Unable to generate insights and perform sentiment analysis.')
