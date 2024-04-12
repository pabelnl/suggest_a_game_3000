import os
import ast
import pandas as pd


# Get rawg api key from enviroment variable
rawg_api_key = os.environ.get('RAWG_API_KEY')
if rawg_api_key: 
    print("API Key loaded successfully!")
else:
    print("Error: RAWG_API_KEY environment variable not found.")
    

# Dataframe format and cleaning

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
                         "platform_name_1", "genre_0", "genre_1"]
    # Iterating thru the columns
    for col in columns_undefined:
        # Checking if dataframe columns contains previous defined column
        if col in df.columns:
            df[col].fillna(value="undefined", inplace=True)
            
    # Filling rating_percentage with 0.00 for nans
    for col in ["rating_percentage_0", "rating_percentage_1", "rating_percentage_2", "rating_percentage_3"]:
        if col in df.columns:
            df[col].fillna(value=0.00, inplace=True)
    

    # TODO: Increase support to more platforms
    # TODO: Increase support to more genres
    # Dropping unnecessary columns    
    df.drop(["platform_name_2","platform_name_3","platform_name_4"], axis=1, inplace=True, errors="ignore")
    df.drop(["genre_2","genre_3","genre_4"], axis=1, inplace=True, errors="ignore")

    # Filter records with Nans for column "released"
    df = df[df["released"].isna() == False]
    # Removing the nan value in the name column
    df = df[df["name"].isna() == False]
    print("* Finished replacing and filtering nans.")
    return df

def clean_format_and_export(df, temporary=True):
    # Dimensionality Reduction, Note: "Unnamed: 0" is added everytime the csv is exported and loaded
    df.drop(["Unnamed: 0", "tba","slug","id","background_image", "dominant_color", "reviews_text_count", "added", "added_by_status",
         "updated","user_game","saturated_color", "dominant_color", "short_screenshots", "parent_platforms", "stores"], axis=1, inplace=True, errors='ignore')
  
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
    df['tags_extracted'] = df['tags_extracted'].replace("[]", "['Singleplayer']")
    
    # Format rating
    df = format_rating(df)
    # Selecting name of the csv to export, if temporary flag is True, the name will change
    file_name1 = f"{len(df)}TEMP_games_clean_formatted_ready_4_clustering.csv" if temporary else f"{len(df)}_games_clean_formatted_ready_4_clustering.csv"
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
        #df_concat = pd.concat([df,platforms[0],platforms[1],platforms[2],platforms[3],platforms[4]], axis=1)
        for platform in platforms:
            df_concat = pd.concat([df,platform], axis=1)
        
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
        #df_concat = pd.concat([df,tags[0],tags[1],tags[2],tags[3],tags[4]], axis=1)
        for tag in tags:
            df_concat = pd.concat([df,tag], axis=1)
        # Removing already normalized "genres_" column
        df_concat.drop([column], axis=1, inplace=True)
        
        return df_concat

    else:
        print("No matching for: ",column)