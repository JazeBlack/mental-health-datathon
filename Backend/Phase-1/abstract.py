import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mental_health.csv')

#Cleaning the texts
import re
import nltk #Helps to get the list of stop words(words that are not relevant for classification (articles the and a, etc...))

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #Takes only the root of the word that indicates what the word means (Example: Loved gets transformed to love and gets classified to positive review)
#Two columns for a very similar word (love and loved): reduces the final dimensions of the sparse matrix: makes learning better/easier

corpus = []
for i in range(0,len(df['text'])): #1000 reviews in the dataset (0-999)
    text  = re.sub('[^a-zA-Z]',' ',df['text'][i]) #Replace non letters by space: (^) --> Not symbol therefore not small a-z and not A-Z either
    text = text.lower() #converting all letters to lowercase
    text = text.split() #splits the review into a list of words
    #Examples of stemming: Running, ran --> run, organization --> organ, etc...
    ps = PorterStemmer()
    text_new = []
    all_stopwords = stopwords.words('english')
    #all_stopwords.remove('not') #removing not from the list of english stopwords
    for word in text:
        if(word not in all_stopwords):
            word = ps.stem(word)
            text_new.append(word)
    text_new = ' '.join(text_new) #Joining the list of strings
    corpus.append(text_new)

#Creating the feature/sparse matrix (CREATING THE BAG OF WORDS MODEL)
#Tokenization: Process of creating the columns corresponding to each of the words of the feature sparse matrix

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000) #Parameter: Max number of columns in each feature vector, approximate down to the nearest whole number to get rid of dummy words
X = cv.fit_transform(corpus).toarray() #.toarray() Converts sparse matrices to dense numpy arrays
#X = X.reshape(-1,1)
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier()
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_test)                     #81.27% accuracy


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)                     #91.05% accuracy

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

def myFun(text):
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text_new = []
    all_stopwords = stopwords.words('english')
    #all_stopwords.remove('not') #removing not from the list of english stopwords
    for word in text:
        if(word not in all_stopwords):
            word = ps.stem(word)
            text_new.append(word)
    text_new = ' '.join(text_new) #Joining the list of strings
    corpus = []
    corpus.append(text_new)
    return cv.transform(corpus).toarray()

    
user_input = input("Enter how you have been feeling as of late: ")
X_test = myFun(user_input)

output1 = classifier.predict(X_test)

def get_output1():
    return int(output1)