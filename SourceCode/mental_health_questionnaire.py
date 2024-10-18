import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_and_train_model():
    """
    Loads the dataset from a CSV file, preprocesses it by removing unnecessary columns,
    splits it into training and testing sets, and trains a Logistic Regression model.

    Returns:
        LogisticRegression model if the CSV is loaded successfully, otherwise None.
    """
    try:
        # Load the dataset
        X = pd.read_csv('MHD_Final_Feature_Engineered.csv')
        
        # Drop unnecessary columns
        X.drop(columns='Unnamed: 0', inplace=True)
        
        # Separate features and target
        X_new = X.iloc[:, :-1]
        y_new = X.iloc[:, -1]
    except FileNotFoundError:
        print("Error: 'MHD_Final_Feature_Engineered.csv' not found. Please check the directory.")
        return None

    # Initialize the Logistic Regression model
    reg = LogisticRegression()

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    
    # Train the model
    reg.fit(X_train, y_train)

    return reg

# Mapping categorical survey responses to numerical values
mappings = {
    'Gender': {'Female': 2, 'Male': 1},
    'Occupation': {'Corporate': 2, 'Student': 3, 'Business': 2, 'Housewife': 3, 'Other': 1},
    'self_employed': {'Yes': 3, 'No': 1},
    'family_history': {'Yes': 3, 'No': 1},
    'treatment': {'Yes': 3, 'No': 1},
    'Days_Indoors': {'1-14 days': 1, 'Go out Every day': 1, 'More than 2 months': 3, '15-30 days': 2, '31-60 days': 2},
    'Growing_Stress': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Changes_Habits': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Mental_Health_History': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Mood_Swings': {'High': 3, 'Medium': 2, 'Low': 1},
    'Coping_Struggles': {'Yes': 3, 'No': 1},
    'Work_Interest': {'Yes': 1, 'Maybe': 2, 'No': 3},
    'Social_Weakness': {'Yes': 3, 'Maybe': 2, 'No': 1},
    'mental_health_interview': {'Yes': 3, 'Maybe': 2, 'No': 1},
    'care_options': {'Yes': 1, 'Not sure': 2, 'No': 3}
}

# Questions to be presented to the user in the survey
questions = [
    {"question": "Enter your gender", "options": ["Male", "Female"]},
    {"question": "What is your occupation?", "options": ["Corporate", "Student", "Business", "Housewife", "Other"]},
    {"question": "Are you self-employed?", "options": ["Yes", "No"]},
    {"question": "Does your family have a history of mental health issues?", "options": ["Yes", "No"]},
    {"question": "Have you ever received treatment or therapy for mental health issues?", "options": ["Yes", "No"]},
    {"question": "How many days do you spend indoors?", 
     "options": ["1-14 days", "Go out Every day", "More than 2 months", "15-30 days", "31-60 days"]},
    {"question": "Is your stress level increasing?", "options": ["Yes", "No", "Maybe"]},
    {"question": "Have you experienced changes in your habits recently?", "options": ["Yes", "No", "Maybe"]},
    {"question": "Do you have a history of mental health issues?", "options": ["Yes", "No", "Maybe"]},
    {"question": "Describe your mood swings", "options": ["High", "Medium", "Low"]},
    {"question": "Do you have difficulty coping?", "options": ["Yes", "No"]},
    {"question": "Are you interested in completing tasks?", "options": ["Yes", "No", "Maybe"]},
    {"question": "Do you struggle to maintain social relationships?", "options": ["Yes", "No", "Maybe"]},
    {"question": "Are you open to a mental health interview?", "options": ["Yes", "No", "Maybe"]},
    {"question": "Are you aware of mental health care options?", "options": ["Yes", "No", "Not sure"]}
]

def map_responses(responses):
    """
    Maps user responses to their corresponding numerical values based on the 'mappings' dictionary.

    Args:
        responses (list): List of responses from the user.

    Returns:
        list: List of numerical values corresponding to the user responses.
    """
    mapped_input = []
    for key, response in zip(mappings.keys(), responses):
        mapped_input.append(mappings[key].get(response, 1))  # Default to 1 if response not found
    return mapped_input

# Load the model once when the module is loaded
model = load_and_train_model()

def get_output2(responses):
    """
    Predicts the mental health condition based on the user responses.

    Args:
        responses (list): User responses collected from the survey.

    Returns:
        int: Predicted class representing the mental health condition, or None if the model is not loaded.
    """
    if model is None:
        return None
    
    # Map the responses to numerical values
    mapped_input = map_responses(responses)
    
    # Make a prediction using the trained model
    prediction = model.predict([mapped_input])
    return int(prediction[0])

def get_questions():
    """
    Retrieves the list of questions to be presented to the user.

    Returns:
        list: List of dictionaries representing questions and their options.
    """
    return questions
