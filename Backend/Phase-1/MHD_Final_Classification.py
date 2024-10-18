#%% Importing necessary libraries
import pandas as pd
import numpy as np

# Load the preprocessed dataset with feature engineering
X = pd.read_csv('MHD_Final_Feature_Engineered.csv')

# Split features (X_new) and target (y_new)
X_new = X.iloc[:, :-1]  # All columns except the last (features)
y_new = X.iloc[:, -1]   # The last column is the target (mental health issue)

# Drop the unnamed column that was added during CSV export (if any)
X_new.drop(columns='Unnamed: 0', inplace=True)

#%% Logistic Regression Model

# Import logistic regression and train-test split
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Train the logistic regression model
reg.fit(X_train, y_train)

# Predict the target for the test data
y_pred = reg.predict(X_test)

#%% Model Evaluation

# Import metrics for evaluation
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred)

#%% Mappings for categorical variables

# This dictionary defines the mappings used to convert categorical values into numerical representations
mappings = {
    'Gender': {'Female': 2, 'Male': 1},
    'Occupation': {'Corporate': 2, 'Student': 3, 'Business': 2, 'Housewife': 3, 'Others': 1},
    'self_employed': {'Yes': 3, 'No': 1},
    'family_history': {'Yes': 3, 'No': 1},
    'treatment': {'Yes': 3, 'No': 1},
    'Days_Indoors': {'1-14 days': 1, 'Go out Every day': 1, 'More than 2 months': 3, '15-30 days': 2, '31-60 days': 2},
    'Growing_Stress': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Changes_Habits': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Mental_Health_History': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Mood_Swings': {'High': 3, 'Low': 1, 'Medium': 2},
    'Coping_Struggles': {'Yes': 3, 'No': 1},
    'Work_Interest': {'No': 3, 'Maybe': 2, 'Yes': 1},
    'Social_Weakness': {'Yes': 3, 'Maybe': 2, 'No': 1},
    'mental_health_interview': {'Yes': 3, 'Maybe': 2, 'No': 1},
    'care_options': {'Yes': 1, 'Not sure': 2, 'No': 3}
}

#%% Questions Template for User Input

# This is a sample list of questions that correspond to the features in the dataset.
'''
1) Gender: Male or Female?
2) Occupation: Are you working in a Corporation, a Student, a Business, a Housewife, or other?
3) Are you self_employed? Yes or No?
4) Does your family have a history of mental health issues? Yes or No?
5) Have you ever had previous treatment or therapy for mental health issues? Yes or No?
6) How many Days do you spend Indoors? 1-14, Go out Every day, More than 2 months, 15-30 days, or 31-60 days?
7) Would you say that your stress level is increasing as of late? Yes, No, or Maybe?
8) Have you experienced any changes in your habits as of late? Yes, No, or Maybe?
9) Have you ever had any previous history of mental health issues? Yes, No, Maybe?
10) How would you describe your mood swings? High, Low, or Medium?
11) Do you have difficulty coping? Yes or No?
12) Are you interested to get work that needs to be done over with? Yes, Maybe, or No?
13) Do you find it difficult to interact with people or maintain social relationships? Yes, Maybe, or No?
14) Are you willing to be interviewed about your mental health? Yes, No, or Maybe?
15) Are you aware of or are using any existing mental health care options/resources? Yes, No, or Not sure?
'''

#%% Sample Input for Classification

# Example input based on the questions above
sample_input = ['Male', 'Student', 'No', 'No', 'No', 'Go out Every day', 'Yes', 'Yes', 'Maybe', 'Medium', 'Yes', 'Maybe', 'Yes', 'Maybe', 'No']

#%% Classification Function

# This function takes a list of categorical inputs, maps them to their corresponding numerical values, and uses the logistic regression model to predict the target.
def classify_target(sample_input):
    keys = mappings.keys()  # Get all the feature names (keys)
    
    # Map the input to numerical values using the predefined mappings
    mapped_input = [mappings[key][value] for key, value in zip(keys, sample_input)]
    
    # Predict the target (mental health issue) based on the mapped input
    return reg.predict([mapped_input])

# Print the predicted result for the sample input
print(classify_target(sample_input))
#%%
