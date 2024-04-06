import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

df = pd.read_csv('data/game_formatted_clean_clustered.csv')

@app.route('/')
def index():
    # Sample questions 
    question1 = "What is your name?"
    question2 = "Which video game platform do you like?"

    return render_template('index.html', question1=question1, question2=question2)

@app.route('/process', methods=['POST'])
def process_form():
    # Retrieve user answers here (you'll need form fields in your HTML)
    user_name = request.form['name'] 
    category = request.form['category']

    # Filter your DataFrame based on the user's input
    filtered_data = df[df['transaction_type'] == category]  # Example filter 

    # Pass the filtered data to the template for display
    return render_template('results.html', data=filtered_data.to_html(), name=user_name)

if __name__ == '__main__':
    app.run(debug=True)