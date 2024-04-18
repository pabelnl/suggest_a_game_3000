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
    filtered_data = df
    
    # Filter your DataFrame based on the user's input
    df.drop(["Unnamed: 0"], axis=1, inplace=True, errors="ignore")
    if " available" in game_name.lower():
        game_name = game_name.lower().split(" available")[0]
        filtered_data = df[df['name'] == game_name]
    else:
        filtered_data = df[df['name'].str.startswith(game_name)]
    
    # Check suggest value and retrieve a game suggestion
    if suggest == "True":
        print("Suggesting a game")
        suggested_game = get_suggestions(df, game_name.lower())
        previous_game_list = ast.literal_eval(previous_game_list)
        
    # Create a name list for every game in the filtered_data
    new_game_list = []
    new_game_platforms_list = []
    
    if len(filtered_data) > 1 or len(filtered_data) == 0:
        if len(filtered_data) == 0:
            filtered_data_ = get_suggestions(df, game_name.lower())
            # If new data is not empty assign it to previous filtered_data and continue the flow
            filtered_data = filtered_data_ if len(filtered_data_) != 0 else filtered_data
            
            
        for i in range(len(filtered_data)):
            game_ = filtered_data.iloc[i].to_frame().T
            # Retrieve available platforms
            available_platforms = get_available_platforms(game_)
            
            obj = {
                "name": game_["name"].iloc[0].title(),
                "platforms": f'{", ".join(available_platforms)}'
            }
            new_game_list.append(f'{obj["name"]} available on: {obj["platforms"]}')
            new_game_platforms_list.append(f'available on: {obj["platforms"]}')
            
    elif len(filtered_data) == 1:
        game_ = filtered_data.iloc[0].to_frame().T
        # Retrieve available platforms
        available_platforms = get_available_platforms(game_)
        
        obj = {
                "name": game_["name"].iloc[0].title(),
                "platforms": f'{", ".join(available_platforms)}'
            }
        new_game_list.append(f'{obj["name"]} available on: {obj["platforms"]}')
        new_game_platforms_list.append(f'available on: {obj["platforms"]}')
        
    # Pass the filtered data to the template for display
    return render_template('results.html', new_game_list = new_game_list, 
                           name = user_name, suggest_boolean = suggest,
                           game_name = game_name,
                           game_platforms_list = new_game_platforms_list,
                           previous_game_list = previous_game_list,
                           suggested_game = suggested_game
                           )

if __name__ == '__main__':
    app.run(debug=True)
