import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Declare output2 globally so it's accessible
output2 = None

def load_and_train_model():
    try:
        X = pd.read_csv('MHD_Feature_Engineered.csv')
        X.drop(columns='Unnamed: 0', inplace=True)
        X_new = X.iloc[:, :-1]
        y_new = X.iloc[:, -1]
    except FileNotFoundError:
        print("Error: 'MHD_Feature_Engineered.csv' not found. Please check the directory.")
        return None

    # Train the logistic regression model
    reg = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    reg.fit(X_train, y_train)

    # Evaluate the model
    y_pred = reg.predict(X_test)
    # print(f"Model F1 Score: {f1_score(y_test, y_pred)}")
    # print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    return reg

# Mappings for responses
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

def ask_question(question, options):
    """Ask a question and validate the user input."""
    options_lower = [opt.lower() for opt in options]  # Convert options to lowercase for comparison
    print(f"{question} (Options: {', '.join(options)})")

    while True:
        response = input().strip().lower()  # User input in lowercase for smooth comparison
        if response in options_lower:
            # Return the original case-sensitive option for mapping
            return options[options_lower.index(response)]
        else:
            print(f"Invalid input. Please choose from: {', '.join(options)}")

# List of questions with options
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
    """Map user responses to numerical values using the mappings dictionary."""
    mapped_input = []
    for key, response in zip(mappings.keys(), responses):
        mapped_input.append(mappings[key].get(response, 1))  # Default to 1 if not found
    return mapped_input

def classify_target(model, responses):
    """Classify the target based on the user's responses."""
    global output2
    try:
        mapped_input = map_responses(responses)
        prediction = model.predict([mapped_input])
        output2 = prediction[0]  # Set global output2
        print(f"Prediction: {output2}")
    except Exception as e:
        print(f"An error occurred during classification: {e}")
        output2 = None  # Ensure output2 is set to None in case of error


def get_output2():
    """Return the value of output2."""
    if output2 is None:
        print("Run the questionnaire first to set output2.")
    return output2

def run_questionnaire():
    """Run the questionnaire to generate output2."""
    model = load_and_train_model()
    if model is None:
        return

    print("\nPlease answer the following questions:\n")
    responses = [ask_question(q['question'], q['options']) for q in questions]

    print("Your responses:", responses)
    classify_target(model, responses)

