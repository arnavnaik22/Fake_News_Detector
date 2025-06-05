from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')  # Only needed once

app = Flask(__name__)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load models
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
lr = pickle.load(open('models/fake_news_model.pkl', 'rb'))
rf = pickle.load(open('models/random_forest_model.pkl', 'rb'))
nb = pickle.load(open('models/naive_bayes_model.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    votes = None
    if request.method == 'POST':
        news = request.form['news_text']
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        preds = [lr.predict(vec)[0], rf.predict(vec)[0], nb.predict(vec)[0]]
        result = 'Real News' if preds.count(1) > 1 else 'Fake News'
        votes = f'LR: {preds[0]}, RF: {preds[1]}, NB: {preds[2]}'
    return render_template('index.html', result=result, votes=votes)

if __name__ == '__main__':
    app.run(debug=True)
