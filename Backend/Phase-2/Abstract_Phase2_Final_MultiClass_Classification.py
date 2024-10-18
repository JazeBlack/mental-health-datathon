import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the balanced dataset from a CSV file, skipping the first column (index)
balanced_df = pd.read_csv('Multi_Class_Main_Final.csv')
balanced_df = balanced_df.iloc[:, 1:]  # Exclude the first column (assumed index)

#%%

# Import necessary libraries for text preprocessing
import re
import nltk

nltk.download('stopwords')  # Download the stopwords list for filtering common words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the 'statement' column is treated as string type
balanced_df['statement'] = balanced_df['statement'].astype(str)

# Preprocess the text data: cleaning, tokenizing, stemming, and removing stopwords
corpus = []  # Initialize an empty list to store the cleaned text data
for i in range(len(balanced_df['statement'])):
    # Remove all non-alphabetic characters, convert to lowercase, and split into words
    text = re.sub('[^a-zA-Z]', ' ', balanced_df['statement'][i])  
    text = text.lower()
    text = text.split()

    ps = PorterStemmer()  # Initialize a PorterStemmer for stemming
    all_stopwords = stopwords.words('english')  # Get the list of English stopwords

    # Stem words and remove stopwords from the text
    text_new = [ps.stem(word) for word in text if word not in all_stopwords]
    text_new = ' '.join(text_new)  # Join the processed words back into a string
    corpus.append(text_new)  # Append the cleaned text to the corpus

#%%

# Creating a Bag of Words model to convert the text into numerical feature vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)  # Limit to the top 5000 most frequent words

# Convert the corpus into a dense array (feature matrix)
X = cv.fit_transform(corpus).toarray()

#%%

# Import the XGBoost classifier for classification
from xgboost import XGBClassifier
classifier = XGBClassifier()  # Initialize the classifier

# The target variable is the 'status' column (last column of the dataframe)
y = balanced_df.iloc[:, -1]

# Split the data into training and test sets (70% training, 30% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the XGBoost classifier on the training data
classifier.fit(X_train, y_train)

#%%

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Import evaluation metrics and compute the F1 score and accuracy
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
f1 = f1_score(y_test, y_pred, average='macro')  # F1 score for multi-class classification
acc = accuracy_score(y_test, y_pred)  # Accuracy of the model

# Print the F1 score and accuracy of the model
print(f"f1 score: {f1: 0.2f}, accuracy: {acc: 0.2f}")
