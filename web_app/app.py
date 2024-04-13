import pandas as pd
from suggest_games import *
from flask import Flask, render_template, request
from module import *
import ast

app = Flask(__name__)

df = pd.read_csv('data/15759_games_clean_formatted_CLUSTERED_RANKED.csv')

@app.route('/')
def index():
    question1 = "What is your name?"
    question2 = "Enter a video game that you like:"

    return render_template('index.html', question1=question1, question2=question2)

@app.route('/process', methods=['POST'])
def process_form():
    # Retrieve user answers here
    user_name = request.form['name']
    game_name = request.form['video_game_name']
    suggest = request.form['suggest_boolean']
    previous_game_list = request.form['previous_game_list']
    suggested_game = "False"
    
    # Filter your DataFrame based on the user's input
    df.drop(["Unnamed: 0"], axis=1, inplace=True, errors="ignore")
    filtered_data = df[df['name'].str.startswith(game_name)]
    
    # Check suggest value
    if suggest == "True":
        print("Suggesting a game")
        suggested_game = get_suggestions(df, game_name.lower())
        previous_game_list = ast.literal_eval(previous_game_list)
        

    # Create a name list for every game in the filtered_data
    new_game_list = []
    if len(filtered_data) > 0:
        for i in range(len(filtered_data)):
            new_game_list.append(filtered_data.iloc[i].to_frame().T["name"].values[0].title())
            
    # Pass the filtered data to the template for display
    return render_template('results.html', new_game_list = new_game_list, 
                           name = user_name, suggest_boolean = suggest,
                           game_name = game_name,
                           previous_game_list = previous_game_list,
                           suggested_game = suggested_game
                           )

if __name__ == '__main__':
    app.run(debug=True)
    #app.run()
