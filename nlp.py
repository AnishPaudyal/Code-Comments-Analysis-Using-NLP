import nltk
import string
import gensim
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import streamlit as st
import re

# Loading the CSV file
data = pd.read_csv('codesnippets.csv')

# Cleaning the column names
data.columns = data.columns.str.strip()

# Extracting relevant columns
comments = data['Message'].tolist()

# Label assignment based on keywords and regex
#for this dataset (imbalance distrbution), so classification may not be refined
def categorize_comment(comment):
    comment_lower = comment.lower()
    if re.search(r"(fix|refactor|debug|optimize|improve|implement|enhance|extend|modify)", comment_lower):
        return "task"
    elif re.search(r"(error|warning|issue|exception|bug)", comment_lower):
        return "warning"
    elif re.search(r"(api|function|class|method|variable|parameter|argument|return|input|output|usage)", comment_lower):
        return "documentation"
    else:
        return "explanation"

labels = [categorize_comment(comment) for comment in comments]

# NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Data Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Only keep alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

processed_comments = [preprocess_text(comment) for comment in comments]

# Word Embeddings using Word2Vec
word2vec_model = Word2Vec(sentences=processed_comments, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(processed_comments, total_examples=len(processed_comments), epochs=10)

# Conversion of each comment into an average of word vectors
def get_comment_vector(comment):
    vector = np.zeros(100)
    count = 0
    for word in comment:
        if word in word2vec_model.wv:
            vector += word2vec_model.wv[word]
            count += 1
    if count > 0:
        vector /= count
    return vector

X = np.array([get_comment_vector(comment) for comment in processed_comments])
y = np.array(labels)

# Spliting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Classification using Random Forest and SVM

# Creating a parameter grid for the SVM classifier
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}

# Grid Search for hyperparameter tuning
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Model Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report:\n", report)

# Cross-Validation
scores = cross_val_score(best_model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {np.mean(scores)}")

# Visualize the classification report using a bar plot
f1_scores = {key: report[key]['f1-score'] for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']}

# Create a figure and axis for the bar plot
fig, ax = plt.subplots()

# Prepare data for plotting
categories = list(f1_scores.keys())
scores = list(f1_scores.values())

# Create the bar plot
ax.bar(categories, scores, color=['blue', 'orange', 'green', 'red'])

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Scores by Category')

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def get_sentiment(comment):
    scores = sia.polarity_scores(comment)
    return "positive" if scores['compound'] > 0 else "negative" if scores['compound'] < 0 else "neutral"

# Streamlit Deployment
st.title("Code Comment NLP Analysis")

# User input
user_input = st.text_area("Enter a code comment to analyze")

if st.button("Analyze"):
    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Check if there are any tokens after preprocessing
    if processed_input:
        # Get the comment vector
        input_vector = get_comment_vector(processed_input).reshape(1, -1)

        # Classification Prediction
        classification_prediction = best_model.predict(input_vector)[0]

        # Sentiment Prediction
        sentiment_prediction = get_sentiment(user_input)

        # Display results
        st.write(f"Classification: {classification_prediction}")
        st.write(f"Sentiment: {sentiment_prediction}")
    else:
        st.write("No valid words in the input. Please enter a valid comment.")

# Display the plot in Streamlit after user interaction
st.subheader("F1 Scores by Category")
st.pyplot(fig)

# Example sentiment analysis on comments
for comment in comments:
    sentiment = get_sentiment(comment)
    print(f"Comment: {comment}, Sentiment: {sentiment}")
