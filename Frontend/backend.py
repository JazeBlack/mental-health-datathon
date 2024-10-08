import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the dataset and initialize the classifier and vectorizer (mimicking the previous training process)
df = pd.read_csv('mental_health.csv')
nltk.download('stopwords')
cv = CountVectorizer(max_features=5000)

# Cleaning and preprocessing the text
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
for i in range(0, len(df['text'])):
    text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    text = text.lower()
    text = text.split()
    text_new = [ps.stem(word) for word in text if word not in all_stopwords]
    corpus.append(' '.join(text_new))

X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

# Train the classifier
classifier = LogisticRegression()
classifier.fit(X, y)

# Prediction function to use in Flask app
def predict_mental_health(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text_new = [ps.stem(word) for word in text if word not in all_stopwords]
    corpus = [' '.join(text_new)]
    X_test = cv.transform(corpus).toarray()
    prediction = classifier.predict(X_test)
    return prediction[0]
