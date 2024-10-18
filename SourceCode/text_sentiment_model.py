import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def prepare_data():
    """
    Prepares the dataset for text classification by:
    1. Loading the data from a CSV file.
    2. Cleaning the text data (removing special characters, converting to lowercase, removing stopwords, and stemming).
    3. Vectorizing the cleaned text using CountVectorizer.
    4. Splitting the data into training and testing sets.
    5. Training a Logistic Regression model on the training set.

    Returns:
        cv (CountVectorizer): The fitted CountVectorizer object for transforming new text.
        classifier (LogisticRegression): The trained Logistic Regression classifier.
    """
    # Load dataset
    df = pd.read_csv('mental_health.csv')
    
    corpus = []
    # Text preprocessing: cleaning, lowercasing, removing stopwords, and stemming
    for text in df['text']:
        text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
        text = text.lower()  # Convert to lowercase
        text = text.split()  # Tokenize words
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')  # Load English stopwords
        text_new = [ps.stem(word) for word in text if word not in all_stopwords]  # Remove stopwords and apply stemming
        corpus.append(' '.join(text_new))  # Re-join words into a single string
    
    # Convert text into feature vectors (Bag of Words model)
    cv = CountVectorizer(max_features=5000)
    X = cv.fit_transform(corpus).toarray()  # Convert corpus to numerical feature vectors
    
    # Extract target labels (assuming last column contains labels)
    y = df.iloc[:, -1].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Logistic Regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    return cv, classifier

# Prepare the data and train the model
cv, classifier = prepare_data()

def predict_sentiment(text):
    """
    Predicts the sentiment or mental health state based on the input text.

    Args:
        text (str): Input text to be classified.

    Returns:
        int: Predicted class label (0 or 1).
    """
    # Preprocess input text (similar to how the training data was preprocessed)
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize words
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')  # Load English stopwords
    text_new = [ps.stem(word) for word in text if word not in all_stopwords]  # Remove stopwords and apply stemming
    text_new = ' '.join(text_new)  # Re-join words into a single string

    # Convert the preprocessed text to feature vector using the trained CountVectorizer
    X_test = cv.transform([text_new]).toarray()

    # Predict the sentiment or class label using the trained classifier
    prediction = classifier.predict(X_test)
    
    return int(prediction[0])

def get_output1(text):
    """
    Wrapper function that predicts the mental health classification for the given text.

    Args:
        text (str): Input text for classification.

    Returns:
        int: Predicted class label.
    """
    return predict_sentiment(text)
