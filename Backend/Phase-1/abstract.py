import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)

def prepare_data():
    df = pd.read_csv('mental_health.csv')
    corpus = []
    for text in df['text']:
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        text_new = [ps.stem(word) for word in text if word not in all_stopwords]
        corpus.append(' '.join(text_new))
    
    cv = CountVectorizer(max_features=5000)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    return cv, classifier

cv, classifier = prepare_data()

def predict_sentiment(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    text_new = [ps.stem(word) for word in text if word not in all_stopwords]
    text_new = ' '.join(text_new)
    X_test = cv.transform([text_new]).toarray()
    prediction = classifier.predict(X_test)
    return int(prediction[0])

def get_output1(text):
    return predict_sentiment(text)
