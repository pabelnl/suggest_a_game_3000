import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
import joblib
from myconfig import *

# load the model from disk
kmeans = joblib.load("15759_games_kmean7.sav")
cluster_centers = kmeans.cluster_centers_

def select_game_from_list(df):
    available_index = []
    
    for i in df.index:
        name = df.loc[i]["name"]
        available_platforms = get_available_platforms(df, i)
        available_index.append(i)
    
        print(f"{i})- {name.title()} available on {', '.join(available_platforms)}")

    game_index = input("Which one is the game?(Enter index number)")
    game_index = int(game_index)
    
    if game_index in available_index:
        selected_game = df.loc[game_index].to_frame().T
        
        available_platforms = get_available_platforms(df, game_index)
        
        print("--------------------------------------------------------------------------------------------")
        print(f'* Selected - {selected_game["name"].str.title()} available on {", ".join(available_platforms)}')
        print("--------------------------------------------------------------------------------------------")
        
        return selected_game
    else:
        print(f"User input is not a valid index")
        return np.nan
    
def get_available_platforms(game, index = 0):
    """Extracts a list of non-empty, non-"undefined" platform names for a given game.

    Searches for platform columns (named 'platform_name_0',  and so on) within a game's 
    dataframe and returns a list of the platform names associated with that game.

    Args:
        game (pandas.Series or pandas.DataFrame): A single game record containing platform information.
        index (int, optional): The index of the game record within a larger DataFrame. 
                               Defaults to 0.

    Returns:
        list: A list of platform names where the game is available, excluding "undefined" entries.
    """
    
    platform_columns = ["platform_name_0", "platform_name_1", "platform_name_2", "platform_name_3", "platform_name_4"]
    available_platforms = []
    
    for col in platform_columns:
        if col in game.columns:
            if game[col].iloc[index] != "undefined":
                available_platforms.append(game[col].iloc[index])
    
    return available_platforms

def format_clean_game(games):
    """Prepares game data by formatting and cleaning.

    Takes a games DataFrame and transforms it into a clean DataFrame ready for analysis and clustering. 
    Leverages the 'clean_format_and_export' function
    for the core transformation process.

    Args:
        games (pandas.DataFrame): The raw game data.

    Returns:
        pandas.DataFrame: A cleaned and formatted DataFrame containing the processed game data.
    """
    df = pd.DataFrame(games)
    df = clean_format_and_export(df)
    
    return df
    
def get_cluster_for_game(games, df):
    """Assigns a game to a cluster and suggests a similar game based on clustering analysis.

    Processes game data, finds the closest cluster based on numerical features, and 
    recommends a random game from the same cluster. Assumes the provided DataFrame 
    ('df') contains pre-calculated cluster assignments and cluster centers.

    Args:
        games (pandas.DataFrame):  Game data to be processed and assigned a cluster.
        df (pandas.DataFrame): A DataFrame containing minimum: 
            * 'cluster': Column indicating cluster assignments for existing games.
            * 'platform_name_0': Column indicating a game's primary platform.
            * 'name': Column containing game names.
            * Numerical feature columns used for clustering (assumed).

    Prints:
        * Platform information for the input game.
        * The assigned cluster number.
        * A recommendation for a similar game (name and platform) from the same cluster.
    """
    processed_games = format_clean_game(games)
    print("---------------------------------------------/----------------------------------------------")
    # Show a list of games and allow the user to select one
    selected_game = select_game_from_list(processed_games)
    
    # Select number dtypes only
    numericals = selected_game.copy().select_dtypes(np.number)
    
    # Scaled df
    new_game_scaled = StandardScaler().fit_transform(numericals)
    # Create a list of available clusters in the dataframe
    clusters = sorted(list(df["cluster"].value_counts().index))
    
    # Create cluster center list
    cluster_center_list = []
    for cluster in clusters:
        c = cluster_centers[cluster]
        cluster_center_list.append(c[cluster])
    
    # Create cluster distance list
    distance_list = []
    for center in cluster_center_list:
        distance_list.append(np.linalg.norm(new_game_scaled - center))
    
    # Get cluster for selected game base on the closest cluster center
    cluster = distance_list.index(min(distance_list))
    print("Game belongs to cluster: ", cluster)
    
    # Retrieve from the top 100 in the cluster a random game
    random_game = get_game_recommendation(df, cluster)
    print("--------------------------------------------------------------------------------------------")
    print(f'We also recommend: {random_game["name"].title()} on {random_game["platform_name_0"]}')

def get_game_recommendation(df, cluster):
    """Recommends a random game from a selected cluster.

    Filters a DataFrame to include only games belonging to the specified cluster, 
    sorts the top 100 games by ranking score (descending), and returns a DataFrame containing 
    data for a randomly chosen game.

    Args:
        df (pandas.DataFrame): The DataFrame containing game information, including:
            * 'cluster': Column indicating cluster assignment.
            * 'ranking_score': Column representing a score for game ranking.
        cluster (int): The numerical identifier of the cluster to recommend from.

    Returns:
        pandas.DataFrame: A DataFrame containing a single row representing the 
                          randomly selected game and its attributes.
    """
    
    df = df[df["cluster"] == cluster]
    top_100 = df.sort_values(by=['ranking_score'], ascending=False).head(100)
    
    # Get a index at random from the suggested songs list
    random_index = random.randint(0, len(top_100) - 1)
    random_game = top_100.iloc[random_index].to_frame().T
    
    return random_game
    

def cluster_search(games_df, df):
    """Takes a selected game and searches a DataFrame for other games belonging to the same cluster.
    Recommends a random game from the matching cluster.

    Args:
        temp_df (pandas.DataFrame): A DataFrame containing game information, including:
            * 'name': Column containing game names.
            * 'cluster': Column indicating cluster assignments.  
        game_selection (str): The name of the selected game.
        df (pandas.DataFrame): The main DataFrame containing cluster assignments and game details, including:
            * 'name': Column containing game names.
            * 'cluster': Column indicating cluster assignments. 
            * 'platform_name_0': Column indicating a game's primary platform. 

    Prints:
        * A recommendation for a similar game (name and platform) from the same cluster.
        * "No suggestions found" if no other games are found within the cluster.
    """
    
    # Retrieve cluster number
    cluster = games_df["cluster"].iloc[0]
    # Get a random game from the top 100 rank for the cluster group
    random_game = get_game_recommendation(df, cluster)
    # Retrieve available platforms
    available_platforms = get_available_platforms(random_game)

    #
    result = {
        "name": random_game["name"].iloc[0],
        "platforms": f'{", ".join(available_platforms)}'
    }

    return result
    
def get_suggestions(df, user_input):
    """Provides an interactive game search and recommendation experience.

    Takes user input, searches for matching games in a DataFrame (df), and either presents
    multiple results for the user to select from or directly initiates a cluster-based search.
    If no matches are found, leverages the RAWG API to find similar games and suggests recommendations.  

    Args:
        df (pandas.DataFrame): The DataFrame containing game information, including:
            * 'name': Column containing game names.
            * 'platform_name_0': Column indicating a game's primary platform. 
            * 'cluster': Column indicating cluster assignments (for recommendations).
        rawg_api_key (str): A valid API key for the RAWG game database (https://rawg.io/apidocs). 

    Requires:
        * The 'requests' library for making API calls.
        * The 'cluster_search' and 'get_cluster_for_game' functions (ensure these are documented). 
    """
    
    if " available" in user_input:
        user_input = user_input.split(" available")[0]
    # Retrieve user selected game from the local df
    games_df = df[df['name'] == user_input].reset_index(drop=True)
    
    # One record scenario
    if len(games_df) == 1:
        result = cluster_search(games_df, df)
        
        return result
    
    # No records scenario
    elif len(games_df) == 0:
        search_params = {
            "key": rawg_api_key,
            "search": user_input
        }
        BASE_URL = "https://api.rawg.io/api/"
        
        response = requests.get(BASE_URL + "games", params=search_params)

        if response.status_code == 200:
            data = response.json()
            print("searching games... returned: ",len(data["results"]), "records")
            games = []
            for g in data["results"]:
                games.append(g)

            get_cluster_for_game(games, df)
            
        elif response.status_code == 502:
            print("502 Bad Gateway Error")
            print("Error response:", response.text)