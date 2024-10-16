from flask import Flask, jsonify, render_template, request, redirect, url_for
import csv
from datetime import datetime
from abstract import get_output1
from mhd import get_output2, get_questions
from final_output import calculate_final_output

app = Flask(__name__)
predictions = []

CSV_FILE = 'user_data.csv'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/learn-more')
def learn_more():
    return render_template('learn_more.html')

@app.route('/user-info', methods=['GET', 'POST'])
def user_info():
    error = None
    name = None
    age = None
    city = None

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        city = request.form.get('city')

        if not age.isdigit() or int(age) < 15:
            error = "Age must be 15 and above."
        else:
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, age, city, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            return redirect(url_for("abstract"))

    return render_template('user_info.html', error=error, name=name, age=age, city=city)

@app.route('/abstract', methods=['GET', 'POST'])
def abstract():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            return render_template('abstract.html', error="Please provide valid input.")
        output1 = get_output1(user_input)  # Get output1 from abstract.py
        predictions.append(output1)  # Store output1 in predictions list

        return redirect(url_for('questionnaire'))
    return render_template('abstract.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    questions = get_questions()  # Retrieve the questionnaire

    if request.method == 'POST':
        if request.is_json:
            # Collect all responses sent in JSON format
            responses = request.get_json()
        else:
            # Fallback for form-data (not JSON)
            responses = [value for key, value in request.form.items()]
        
        params = [value for key,value in responses.items()]

        # Use the collected responses for predictions
        output2 = get_output2(params)
        predictions.append(output2)
        final_output = calculate_final_output(predictions[-2], predictions[-1])
        predictions.append(final_output)

        # Determine the result URL based on final output 
        result_url = url_for('result', outcome=final_output)

        # Render the loading screen and then redirect to the result
        return jsonify({'redirect_url': result_url})
    
    # If it's a GET request, render the questionnaire
    return render_template('questionnaire.html', questions=questions)

@app.route('/result/<outcome>')
def result(outcome):
    if outcome == '1':
        return f"Based on our assessment, you may benefit from speaking with a mental health professional.\nOutput 1: {predictions[-3]}\nOutput 2: {predictions[-2]}\nFinal Output: {predictions[-1]}"
    else:
        return redirect(url_for('no_issue'))
    

@app.route('/no_issue')
def no_issue():
    return render_template('no_issue.html')
    
if __name__ == '__main__':
    app.run(debug=True)
