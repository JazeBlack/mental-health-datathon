from flask import Flask, render_template, request, redirect, url_for
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
    sex = None

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        sex = request.form.get('sex')

        if not age.isdigit() or int(age) < 15:
            error = "Age must be 15 and above."
        else:
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, age, sex, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            return redirect(url_for("abstract"))

    return render_template('user_info.html', error=error, name=name, age=age, sex=sex)

@app.route('/abstract', methods=['GET', 'POST'])
def abstract():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            return render_template('abstract.html', error="Please provide valid input.")
        output1 = get_output1(user_input)
        predictions.append(output1)  # Store output1 in session
        return redirect(url_for('questionnaire'))
    return render_template('abstract.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    
    questions = get_questions()  # Retrieve the questionnaire

    if request.method == 'POST':
        responses = []
        for question in questions:
            response = request.form.get(question['question'])
            responses.append(response)
        
        output2 = get_output2(responses)  # Calculate probability from responses
        final_output = calculate_final_output(predictions[0], float(output2))  # Combine both probabilities

        # Show the loading screen with a delay before showing the result
        if final_output == 1:
            result_url = url_for('result', outcome='professional')
        else:
            result_url = url_for('result', outcome='well')
        
        # Redirect to loading screen with a delay
        return render_template('loading.html', redirect_url=result_url, redirect_after=3000)
    
    return render_template('questionnaire.html', questions=questions)

@app.route('/result/<outcome>')
def result(outcome):
    if outcome == 'professional':
        return "Based on our assessment, you may benefit from speaking with a mental health professional."
    else:
        return "Based on our assessment, you appear to be managing well. However, if you have any concerns, don't hesitate to seek professional advice."

if __name__ == '__main__':
    app.run(debug=True)
