from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

app = Flask(__name__)

# Function to preprocess text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Function to perform sentiment analysis using VADER
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    # Get sentiment scores
    sentiment_scores = sid.polarity_scores(text)
    # Classify sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Function to fine-tune GPT model on investment dataset
def fine_tune_gpt(dataset):
    # Load pre-trained GPT-2 model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a new padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize and encode the investment dataset
    input_ids = tokenizer.batch_encode_plus(dataset['Sentiment'].tolist(), padding=True, return_tensors="pt")["input_ids"]

    # Fine-tuning parameters
    learning_rate = 5e-5
    num_epochs = 3
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(input_ids) * num_epochs)

    # Fine-tune the GPT model
    for epoch in range(num_epochs):
        model.train()
        for batch in input_ids:
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            return model, tokenizer

    return model, tokenizer

# Load dataset for fine-tuning GPT model
# Replace 'your_dataset.csv' with your actual dataset file path
dataset = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/training_model.csv')
dataset['Sentiment'] = dataset['News Headline'].apply(preprocess_text)

# Fine-tune GPT model
gpt_model, tokenizer = fine_tune_gpt(dataset)

@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    if request.method == 'POST':
        # Get news headline from the request
        headline = request.json.get('headline', '')

        # Preprocess the news headline
        preprocessed_headline = preprocess_text(headline)

        # Perform sentiment analysis
        sentiment = analyze_sentiment(preprocessed_headline)

        return jsonify({'headline': headline, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)