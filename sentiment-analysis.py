# Sentiment Analysis with Python
# File: sentiment_analyzer.py

import numpy as np
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=1500)
        self.classifier = MultinomialNB()
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of given text using TextBlob
        Returns: Positive, Negative, or Neutral
        """
        analysis = TextBlob(text)
        
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    
    def train_model(self, X_train, y_train):
        """
        Train the model using Naive Bayes classifier
        """
        # Transform the text data into vectors
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Train the classifier
        self.classifier.fit(X_train_vectorized, y_train)
    
    def predict(self, text):
        """
        Predict sentiment using trained model
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Transform the text using fitted vectorizer
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.classifier.predict(text_vectorized)
        
        return prediction[0]

# Example usage
if __name__ == "__main__":
    # Sample dataset
    data = {
        'text': [
            "I love this product, it's amazing!",
            "This is the worst experience ever.",
            "The service was okay, nothing special.",
            "Great customer support and quick delivery!",
            "Disappointed with the quality."
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Train model
    analyzer.train_model(X_train, y_train)
    
    # Test with new text
    test_text = "The product exceeded my expectations!"
    print(f"Text: {test_text}")
    print(f"Predicted Sentiment: {analyzer.predict(test_text)}")
    print(f"TextBlob Sentiment: {analyzer.analyze_sentiment(test_text)}")
