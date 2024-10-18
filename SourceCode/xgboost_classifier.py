# Import necessary libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def load_and_preprocess_data():
    """
    Loads and preprocesses the dataset for text classification.
    
    Returns:
    tuple: Feature matrix X, target variable y, and the fitted CountVectorizer.
    """
    # Load the balanced dataset from the CSV file, skipping the first column (assumed index)
    balanced_df = pd.read_csv('Multi_Class_Main_Final.csv')
    balanced_df = balanced_df.iloc[:, 1:]  # Exclude the first column (index)

    # Ensure 'statement' column is treated as string
    balanced_df['statement'] = balanced_df['statement'].astype(str)

    # Initialize the PorterStemmer and load stopwords
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')

    # Preprocess text data
    corpus = []
    for text in balanced_df['statement']:
        text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        text_new = [ps.stem(word) for word in text if word not in all_stopwords]
        corpus.append(' '.join(text_new))  # Join cleaned words

    # Create Bag of Words model with top 5000 frequent words
    cv = CountVectorizer(max_features=5000)
    X = cv.fit_transform(corpus).toarray()  # Convert text to feature matrix

    # Target variable is the 'status' column
    y = balanced_df.iloc[:, -1]

    return X, y, cv

def train_model(X, y):
    """
    Trains the XGBoost classifier on the given dataset.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (pandas.Series): Target variable.

    Returns:
    XGBClassifier: Trained model.
    """
    # Split the data into training and test sets (70% training, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the XGBoost classifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    return classifier

def preprocess_text(text, cv):
    """
    Preprocesses a user input string for prediction.
    
    Parameters:
    text (str): User input text.
    cv (CountVectorizer): Fitted CountVectorizer.

    Returns:
    numpy.ndarray: Transformed feature vector for the input text.
    """
    # Clean and preprocess the input text
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()

    # Apply stemming and remove stopwords
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    text_new = [ps.stem(word) for word in text if word not in all_stopwords]

    # Join the cleaned text
    text_cleaned = ' '.join(text_new)

    # Transform the user input text into the same Bag of Words format
    return cv.transform([text_cleaned]).toarray()

# Load and prepare the data
X, y, cv = load_and_preprocess_data()

# Train the classifier
classifier = train_model(X, y)

def predict_mental_health(user_input):
    """
    Predicts the mental health condition based on user input.
    
    Parameters:
    user_input (str): User's input text.

    Returns:
    str: Predicted mental health condition.
    """
    X_input = preprocess_text(user_input, cv)
    prediction = classifier.predict(X_input)
    return prediction  # Return the predicted label

def soft_voting_probabilities(responses):
    """
    Combines the probabilities from multiple user responses to predict the final class.
    
    Parameters:
    responses (list of str): List of user input responses.
    
    Returns:
    int: Final predicted class based on combined probabilities.
    """
    # Convert responses into feature vectors
    response_vectors = cv.transform(responses).toarray()

    # Get class probabilities for each response
    probabilities = [classifier.predict_proba([vec])[0] for vec in response_vectors]

    # Sum the probabilities across all responses
    total_prob = np.sum(probabilities, axis=0)

    # Get the class with the highest total probability
    final_class = np.argmax(total_prob)

    return final_class
