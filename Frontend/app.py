from flask import Flask, render_template, request, redirect, url_for
import csv
from datetime import datetime
from backend import predict_mental_health

app = Flask(__name__)

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

        # Check if age is valid
        if not age.isdigit() or int(age) < 15:
            error = "Age must be 15 and above."
        else:
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, age, sex, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            return redirect(url_for("abstract"))
    else:
        return render_template('user_info.html')

    return render_template('user_info.html', error=error, name=name, age=age, sex=sex)

@app.route('/abstract', methods=['GET', 'POST'])
def abstract():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
    
        prediction = predict_mental_health(user_input)

        if prediction == 1:
            return redirect(url_for('questionnaire'))
        else:
            return "Nah fam u good"
    return render_template('abstract.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')


if __name__ == '__main__':
    app.run(debug=True)
