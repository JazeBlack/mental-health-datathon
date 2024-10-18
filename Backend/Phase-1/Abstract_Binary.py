import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
df = pd.read_csv('mental_health.csv')

#%% Text Cleaning and Preprocessing

# Import necessary NLP libraries
import re
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize an empty list to store the cleaned text data (corpus)
corpus = []

# Iterate through the 'text' column in the DataFrame to clean and preprocess each row
for i in range(0, len(df['text'])):
    # Remove all non-alphabet characters and replace them with a space
    text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    
    # Convert all text to lowercase
    text = text.lower()
    
    # Split the text into individual words (tokenization)
    text = text.split()
    
    # Initialize the PorterStemmer to perform stemming (reduce words to their root form)
    ps = PorterStemmer()
    
    # Filter out stopwords (common words with little predictive value) and stem the rest
    text_new = []
    all_stopwords = stopwords.words('english')
    
    for word in text:
        if word not in all_stopwords:
            word = ps.stem(word)
            text_new.append(word)
    
    # Join the cleaned and stemmed words back into a single string
    text_new = ' '.join(text_new)
    
    # Append the cleaned text to the corpus
    corpus.append(text_new)

#%% Feature Extraction - Creating the Bag of Words Model

# Convert the cleaned corpus into a sparse matrix of word counts (Bag of Words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)  # Limit the number of features to 5000 most frequent words
X = cv.fit_transform(corpus).toarray()  # Convert sparse matrix to dense numpy array

# Define the target variable (last column of the DataFrame)
y = df.iloc[:, -1].values

#%% Train-Test Split

# Split the dataset into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Model Training and Evaluation

# Logistic Regression model for text classification
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model using confusion matrix, accuracy, and F1 score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#%% Real-Time User Input Prediction Function

# Function to preprocess and predict mental health condition from user input text
def myFun(text):
    # Clean and preprocess the input text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    
    # Apply stemming and remove stopwords
    ps = PorterStemmer()
    text_new = []
    all_stopwords = stopwords.words('english')
    for word in text:
        if word not in all_stopwords:
            word = ps.stem(word)
            text_new.append(word)
    
    # Convert the cleaned text back into a format that the model can understand
    text_new = ' '.join(text_new)
    corpus = []
    corpus.append(text_new)
    
    # Transform the user input text into the same Bag of Words format
    return cv.transform(corpus).toarray()

#%% Interactive Prediction Loop

# Continuously prompt the user for input and predict the mental health outcome
while True:
    user_input = input("Enter how you have been feeling as of late: ")
    
    # Preprocess the input and make a prediction
    X_test = myFun(user_input)
    print(classifier.predict(X_test))
