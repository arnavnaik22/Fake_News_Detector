
Fake News Detection Flask App
This is a simple web application built with Flask that lets you check whether a news article is likely real or fake. 
It uses machine learning models trained on news data to analyze the input text and predict its authenticity.

Features

Clean and preprocess input news text.

Uses three ML models — Logistic Regression, Random Forest, and Naive Bayes — to predict news authenticity.

Majority vote from models determines the final result.

Easy-to-use web interface where you can paste news text and get instant feedback.

Lightweight and simple code, ideal for learning Flask and ML deployment.

How It Works

You enter or paste a news article in the text box.

The app cleans the text by removing URLs, special characters, and stopwords, then stems the words.

The cleaned text is transformed using a TF-IDF vectorizer.

Three different ML models predict if the news is real or fake.

The final output is decided by majority vote and shown on the webpage.
