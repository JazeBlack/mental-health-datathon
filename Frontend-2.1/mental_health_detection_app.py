from flask import Flask, jsonify, render_template, request, redirect, url_for
import csv
from datetime import datetime
from text_sentiment_model import get_output1
from mental_health_questionnaire import get_output2, get_questions
from final_output_calculator import calculate_final_output
from xgboost_classifier import soft_voting_probabilities

app = Flask(__name__)

# Global variables to store predictions and user responses
predictions = []
response_storage = []

# File to store user data
CSV_FILE = 'user_data.csv'

# Route to the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to the "Learn More" page
@app.route('/learn-more')
def learn_more():
    return render_template('learn_more.html')

# Route to collect basic user info
@app.route('/user-info', methods=['GET', 'POST'])
def user_info():
    error = None
    name = None
    age = None
    city = None

    # Handling POST request when the form is submitted
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        city = request.form.get('city')

        # Validating age input (must be a digit and 15 or older)
        if not age.isdigit() or int(age) < 15:
            error = "Age must be 15 and above."
        else:
            # Write user info to CSV file with timestamp
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, age, city, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            return redirect(url_for("abstract"))

    # Render user info form with any errors
    return render_template('user_info.html', error=error, name=name, age=age, city=city)

# Route for the abstract input stage
@app.route('/abstract', methods=['GET', 'POST'])
def abstract():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        
        # Validate input
        if not user_input:
            return render_template('abstract.html', error="Please provide valid input.")
        
        # Process abstract input using the get_output1 function
        output1 = get_output1(user_input)
        predictions.append(output1)  # Store the result in predictions list

        # Redirect to the questionnaire page
        return redirect(url_for('questionnaire'))

    # Render abstract input form
    return render_template('abstract.html')

# Route for the questionnaire stage
@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    questions = get_questions()  # Retrieve the list of questions for the survey

    if request.method == 'POST':
        # Check if responses are sent as JSON
        if request.is_json:
            responses = request.get_json()  # Get responses as JSON
        else:
            # Fallback for form-data (non-JSON)
            responses = [value for key, value in request.form.items()]
        
        # Process responses and collect the parameters
        params = [value for key, value in responses.items()]
        
        # Use the collected responses for prediction
        output2 = get_output2(params)
        predictions.append(output2)  # Store the result in predictions list
        
        # Combine abstract and questionnaire predictions
        final_output = calculate_final_output(predictions[-2], predictions[-1])
        predictions.append(final_output)  # Store final output
        
        # Determine the appropriate result URL based on the final output
        result_url = url_for('result', outcome=final_output)

        # Return a JSON response for redirection
        return jsonify({'redirect_url': result_url})

    # Render the questionnaire form
    return render_template('questionnaire.html', questions=questions)

# Route to handle the final result and redirect accordingly
@app.route('/result/<outcome>')
def result(outcome):
    # If outcome indicates an issue, redirect to issue_detected page
    if outcome == '1':
        return redirect(url_for('issue_detected'))
    # Otherwise, redirect to no_issue page
    else:
        return redirect(url_for('no_issue'))

# Route to display the issue detected page
@app.route('/issue_detected')
def issue_detected():
    return render_template('issue_detected.html')

# Second abstract phase to handle multiple responses
@app.route('/abstract2', methods=['GET', 'POST'])
def abstract2():
    global response_storage  # Access global storage for user responses

    if request.method == 'POST':
        # Parse incoming JSON data
        data = request.get_json()
        user_responses = data.get('user_input', [])

        # Validate responses (at least 3 valid responses required)
        if not user_responses or len(user_responses) < 3:
            return jsonify({'error': 'At least three valid responses are required'}), 400

        # Store non-empty responses
        response_storage.extend([resp for resp in user_responses if resp.strip()])

        # Once we have at least 3 valid responses, proceed with prediction
        if len(response_storage) >= 3:
            final_prediction = soft_voting_probabilities(response_storage)  # Get the final prediction

            # Reset the storage for future inputs
            response_storage = []

            # Return a JSON response with redirect URL for the final result
            return jsonify({'redirect_url': url_for('prediction_result', result=final_prediction)})

        return jsonify({'error': 'Failed to process the inputs.'}), 400

    # Render the abstract2 input form
    return render_template('abstract2.html')

# Route to display the final prediction result
@app.route('/prediction_result')
def prediction_result():
    # Retrieve result passed as a query parameter
    result = request.args.get('result', type=int)

    if result is None:
        return "Error: Invalid result", 400

    # Map result to mental health condition
    if result == 0:
        my_value = "Anxiety"
    elif result == 1:
        my_value = "Depression"
    elif result == 2:
        my_value = "Suicidal"
    elif result == 3:
        my_value = "Bipolar"
    else:
        return "Error: Invalid prediction value", 400

    # Render the result in the HTML template
    return render_template('prediction_result.html', value=my_value)

# Route for no issue detected page
@app.route('/no_issue')
def no_issue():
    return render_template('no_issue.html')

# Main entry point of the Flask app
if __name__ == '__main__':
    app.run(debug=True)
