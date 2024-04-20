import pandas as pd
import os
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
import joblib

# Get rawg api key from enviroment variable
rawg_api_key = os.environ.get('RAWG_API_KEY')
if rawg_api_key: 
    print("API Key loaded successfully!")
else:
    print("Error: RAWG_API_KEY environment variable not found.")


# load the model from disk
kmeans = joblib.load("15759_games_kmean7.sav")
cluster_centers = kmeans.cluster_centers_


def get_exceptional_rating(row):
    """Searches for an 'exceptional' rating and returns the corresponding percentage.

    Iterates through columns named 'rating_title_0' to 'rating_title_3' in the provided
    row. If a column's value is 'exceptional', the corresponding value from a 
    'rating_percentage_{i}' column is returned. Otherwise, returns 0.

    Args:
        row (pandas.Series): A single row of data from a DataFrame.

    Returns:
        float: The rating percentage associated with the first 'exceptional' rating found,
               or 0 if no 'exceptional' rating is present.
    """
    for i in range(4):
        if row[f"rating_title_{i}"] == "exceptional":
            return row[f"rating_percentage_{i}"]
    return 0

def get_recommended_rating(row):
    """Searches for an 'recommended' rating and returns the corresponding percentage.

    Iterates through columns named 'rating_title_0' to 'rating_title_3' in the provided
    row. If a column's value is 'recommended', the corresponding value from a 
    'rating_percentage_{i}' column is returned. Otherwise, returns 0.

    Args:
        row (pandas.Series): A single row of data from a DataFrame.

    Returns:
        float: The rating percentage associated with the first 'recommended' rating found,
               or 0 if no 'recommended' rating is present.
    """
    for i in range(4):
        if row[f"rating_title_{i}"] == "recommended":
            return row[f"rating_percentage_{i}"]
    return 0

def get_meh_rating(row):
    """Searches for an 'meh' rating and returns the corresponding percentage.

    Iterates through columns named 'rating_title_0' to 'rating_title_3' in the provided
    row. If a column's value is 'meh', the corresponding value from a 
    'rating_percentage_{i}' column is returned. Otherwise, returns 0.

    Args:
        row (pandas.Series): A single row of data from a DataFrame.

    Returns:
        float: The rating percentage associated with the first 'meh' rating found,
               or 0 if no 'meh' rating is present.
    """
    for i in range(4):
        if row[f"rating_title_{i}"] == "meh":
            return row[f"rating_percentage_{i}"]
    return 0
    
def get_skip_rating(row):
    """Searches for an 'skip' rating and returns the corresponding percentage.

    Iterates through columns named 'rating_title_0' to 'rating_title_3' in the provided
    row. If a column's value is 'skip', the corresponding value from a 
    'rating_percentage_{i}' column is returned. Otherwise, returns 0.

    Args:
        row (pandas.Series): A single row of data from a DataFrame.

    Returns:
        float: The rating percentage associated with the first 'skip' rating found,
               or 0 if no 'skip' rating is present.
    """
    for i in range(4):
        if row[f"rating_title_{i}"] == "skip":
            return row[f"rating_percentage_{i}"]
    return 0

# Working with tags column, we will be only extracting if the game is singleplayer or multiplayer
def extract_by_id(data_list, target_ids=[7, 31]):
    """Extracts items from a list of data items based on their IDs.

    Filters a list of dictionaries and returns a list with the 'name' 
    value of any items whose 'id' key matches a value within the 'target_ids' list.  

    Args:
        data_list (list): A list of dictionaries containing 'id' and 'name' keys.
        target_ids (list, optional): A list of target IDs to filter by. Defaults to [7, 31]
        which are SinglePlayer and Multiplayer tags.

    Returns:
        list: A list of 'name' values from the items in 'data_list' whose 'id' matched 
              a value in 'target_ids'.
    """
    result = []
    for item in data_list:
        if item.get("id") in target_ids:
            result.append(item["name"])
    return result
    
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

def fill_na_for(df):
    """ Fill na values with undefined or 0.00 for columns:
        "rating_title_0", "rating_title_1", "rating_title_2", "rating_title_3"
        "rating_percentage_0", "rating_percentage_1", "rating_percentage_2", "rating_percentage_3"
        "platform_name_0", "platform_name_1", "platform_name_2","platform_name_3","platform_name_4"
    Args:
        df (Pandas.Dataframe): Dataframe object.
        
    Returns:
        df (Pandas.Dataframe): Dataframe without nans for the mentioned columns
    """
    
    # Filling rating_title, genre and platform_name with undefined for nans
    columns_undefined = ["rating_title_0", "rating_title_1", "rating_title_2", "rating_title_3", "platform_name_0", 
                         "platform_name_1", "platform_name_2", "platform_name_3", "platform_name_4", "genre_0", "genre_1"]
    # Iterating thru the columns
    for col in columns_undefined:
        # Checking if dataframe columns contains previous defined column
        if col in df.columns:
            df[col].fillna(value="undefined", inplace=True)
            
    # Filling rating_percentage with 0.00 for nans
    for col in ["rating_percentage_0", "rating_percentage_1", "rating_percentage_2", "rating_percentage_3"]:
        if col in df.columns:
            df[col].fillna(value=0.00, inplace=True)
    
    # TODO: Increase support to more genres
    # Dropping unnecessary columns
    df.drop(["genre_2","genre_3","genre_4"], axis=1, inplace=True, errors="ignore")

    # Filter records with Nans for column "released"
    df = df[df["released"].isna() == False]
    # Removing the nan value in the name column
    df = df[df["name"].isna() == False]
    print("* Finished replacing and filtering nans.")
    
    return df

def string_to_object(df, column):
    if df[column].isna().sum() > 0:
        df = df[df[column].isna() == False]
        df.reset_index(drop=True)
    
    if type(df[column].iloc[0]) == str:
        df[column+"_"] = list(map(lambda x: ast.literal_eval(x), df[column]))
        df.drop([column], axis=1, inplace=True)      
    else:
        df[column+"_"] = df[column]
        df.drop([column], axis=1, inplace=True)  
    return df

def json_normalize(df, column):
    df_normalized = pd.json_normalize(df[column])
    result = []
    for x in df_normalized.columns:
        result.append(pd.json_normalize(df_normalized[x]))
    
    # Scenario for "ratings_"
    if column == "ratings_":
        # Removing unnecessary columns for each df
        tags = []
        counter = 0
        for i in result:
            i.drop(["id","count"], axis=1, inplace=True)
            # Renaming columns
            i.columns = [f"rating_title_{counter}", f"rating_percentage_{counter}"]
            # Appending the modified column to ratings variable
            tags.append(i)
            # incrementing the counter
            counter = counter + 1
        # Concatenate the main dataframe with the ratings dfs
        df_concat = pd.concat([df,tags[0],tags[1],tags[2],tags[3]], axis=1)
        # Removing already normalized "ratings_" column
        df_concat.drop([column], axis=1, inplace=True)
        
        return df_concat
    # Scenario for "platforms_"
    elif column == "platforms_":
        # Removing unnecessary columns for each df
        platforms = []
        counter = 0
        for i in result:
            i = i[["platform.name"]]
            # Renaming columns
            i.columns = [f"platform_name_{counter}"]
            # Appending the modified column to platforms variable
            platforms.append(i)
            # incrementing the counter
            counter = counter + 1
            
        # Concatenate the main dataframe with the platform dfs
        df_concat = df
        for platform in platforms:
            df_concat = pd.concat([df_concat,platform], axis=1)
        
        # Removing already normalized "platforms_" column
        df_concat.drop([column], axis=1, inplace=True)
        
        return df_concat
    
    # Scenario for "genres_"
    elif column == "genres_":
        # Removing unnecessary columns for each df
        tags = []
        counter = 0
        for i in result:
            i = i[["name"]]
            # Renaming columns
            i.columns = [f"genre_{counter}"]
            # Appending the modified column to genres variable
            tags.append(i)
            # incrementing the counter
            counter = counter + 1
            
        # Concatenate the main dataframe with the genre dfs
        df_concat = df
        for tag in tags:
            df_concat = pd.concat([df_concat,tag], axis=1)
        # Removing already normalized "genres_" column
        df_concat.drop([column], axis=1, inplace=True)
        
        return df_concat

    else:
        print("No matching for: ",column)

def clean_format_and_export(df, temporary=True):
    # Dimensionality Reduction, Note: "Unnamed: 0" is added everytime the csv is exported and loaded
    df.drop(["Unnamed: 0", "tba","slug","id", "dominant_color", "reviews_text_count", "added", "added_by_status",
         "updated","user_game","saturated_color", "dominant_color", "short_screenshots", "parent_platforms", "stores"], axis=1, inplace=True, errors="ignore")
  
    # First Filter
    # - Excluding the records without any "ratings" and "rating" value 0
    df = df[(df["ratings"].str.len() > 0) & (df["rating"] > 2)]
    df.reset_index(drop=True)
    print("* Applied first filter, new shape: ", df.shape)
    # Change column "released" to datetime
    df["released"] = pd.to_datetime(df["released"], format="%Y-%m-%d")

    # We create a list of columns that needs json normalize
    columns_to_normalize = ["ratings", "platforms", "genres"]
    
    # Iterating thru the list, changing string to object if necessary
    for column in columns_to_normalize:
        if column in df.columns:
            # Peform string_to_object
            df = string_to_object(df, column)
            # Peform json normalize
            df = json_normalize(df, column+"_")
    
    # Extracting singleplayer and multipler tags only
    df = string_to_object(df, "tags")
    df["tags_extracted"] = df["tags_"].apply(extract_by_id)
    df.drop(["tags_"], axis=1, inplace=True)
    print("* Completed string to object and json normalize operations \o/ \o/")
    
    # Dropping "esrb_rating", "clip", "community_rating", "metacritic" half the data has nans and the column constribution is very low for our purposes
    columns_to_drop = ["esrb_rating","clip","community_rating", "metacritic"]
    for column in columns_to_drop:
        if column in df.columns:  
            df.drop([column], axis=1, inplace=True)
    # TODO: Properly fix, meanwhile doing workaroud for known issue
    # Dropping extra plaftorm and genre columns
    platform_cols_to_drop = [f"platform_name_{i}" for i in range(7, 22)]
    df.drop(columns=platform_cols_to_drop, inplace=True, errors='ignore')
    
    genre_cols_to_drop = [f"genre_{i}" for i in range(2, 22)]
    df.drop(columns=genre_cols_to_drop, inplace=True, errors='ignore')
    
    # Perform fill nan values for predefined columns
    df = fill_na_for(df)
    
    # Export the records to a csv
    file_name1 = f"{len(df)}_TEMP_games_formatted_clean.csv" if temporary else f"{len(df)}_games_formatted_clean.csv"
    df.to_csv(file_name1)
    print("* Created export csv file named: ", file_name1)
    
    # Read from csv
    df = pd.read_csv(file_name1)
    
    # Remove unnamed column
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    # Replace empty tag with singleplayer, games should let be at least 1 singleplayer
    df["tags_extracted"] = df["tags_extracted"].replace("[]", "['Singleplayer']")
    
    # Format rating
    df = format_rating(df)
    # Selecting name of the csv to export, if temporary flag is True, the name will change
    file_name1 = f"{len(df)}_TEMP_games_clean_formatted_ready_4_clustering.csv" if temporary else f"{len(df)}_games_clean_formatted_ready_4_clustering.csv"
    print("* Created export csv file ready for clustering named: ", file_name1)
    df.to_csv(file_name1)
    
    return df

def format_rating(df):
    # We are merging the title and rating columns for the each respective rows
    df["exceptional_"] = df.apply(get_exceptional_rating, axis=1)
    df["recommended_"] = df.apply(get_recommended_rating, axis=1)
    df["meh_"] = df.apply(get_meh_rating, axis=1)
    df["skip_"] = df.apply(get_skip_rating, axis=1)
    
    # Dropping rating_title and rating_percentage columns
    df.drop(["rating_title_0","rating_title_1","rating_title_2","rating_title_3", 
                 "rating_percentage_0","rating_percentage_1","rating_percentage_2", "rating_percentage_3"],axis=1, inplace=True)
    print("* Finished formatting rating columns")
    
    return df

def optimize_dtypes(df):
    """Attempts to optimize datatypes in a DataFrame based on analysis.

    Args:
        df (pandas.DataFrame): The DataFrame to optimize.

    Returns:
        pandas.DataFrame: The DataFrame with optimized datatypes.
    """

    df_optimized = df.copy()

    for col in df_optimized.columns:
        current_dtype = df_optimized[col].dtype

        if col not in ["name", "released", "tags_extracted", "platform_name_0", "platform_name_1", "platform_name_2", "platform_name_3", "platform_name_4"]:
            if current_dtype == "object":
                try:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], errors="coerce")
                except ValueError:
                    try:
                        #TODO Remove datetime scenario
                        df_optimized[col] = pd.to_datetime(df_optimized[col], errors="coerce")
                    except ValueError:
                        pass

    return df_optimized

def get_cluster_for_game(games, df, searched_api = False):
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
    games = pd.DataFrame(games)
    processed_games = clean_format_and_export(games)
    
    # Try to change columns to proper dtype
    processed_games = optimize_dtypes(processed_games)
    
    # Select number dtypes only
    numericals = processed_games.copy().select_dtypes(np.number)
    
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
    game_cluster_list = []
    
    if len(new_game_scaled) > 1 and searched_api == True:
        
        for item in new_game_scaled:
            game_distance_list = []
            
            for center in cluster_center_list:
                game_distance_list.append(np.linalg.norm(item - center))
            
            # Get cluster for game base on the closest cluster center
            cluster = game_distance_list.index(min(game_distance_list))
            game_cluster_list.append(cluster)
            

        # Return df with cluster column
        processed_games["cluster"] = game_cluster_list
        
        return processed_games
        
    for center in cluster_center_list:
        distance_list.append(np.linalg.norm(new_game_scaled - center))
    
    # Get cluster for selected game base on the closest cluster center
    cluster = distance_list.index(min(distance_list))
    
    # Retrieve from the top 100 in the cluster a random game
    random_game = get_game_recommendation(df, cluster)
    
    return random_game

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
        "platforms": f'{", ".join(available_platforms)}',
        "img_url": random_game["background_image"].iloc[0]
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
    games_df = df[df["name"] == user_input].reset_index(drop=True)
    
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
            print("* Searching games... returned: ",len(data["results"]), "records")
            
            # Return original df if api call returned 0 records
            if len(data["results"]) == 0:
                return df
            
            # Loop the results and add to games list
            games = []
            for g in data["results"]:
                games.append(g)

            result = get_cluster_for_game(games, df, True)
            
            return result
            
        elif response.status_code == 502:
            print("502 Bad Gateway Error")
            print("Error response:", response.text)