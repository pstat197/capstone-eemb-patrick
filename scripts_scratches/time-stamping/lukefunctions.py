import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from itertools import combinations
pd.options.mode.chained_assignment = None

def key_values(dataframe):
    df_storage = []


    for identity in dataframe["id"].unique():
        df_id = dataframe[dataframe["id"] == identity]

        dx = df_id['pos_x'] - df_id['pos_x'].shift(2)
        dy = df_id['pos_y'] - df_id['pos_y'].shift(2)
        df_id['speed'] = np.sqrt(dx**2 + dy**2)
        df_id = df_id.fillna(0)
        df_id['cum_dist'] = df_id['speed'].cumsum()
        df_id = df_id.sort_values(by=['frame'])

        important_vals = pd.DataFrame()
        important_vals["ID"] = [df_id["id"].iloc[0]]
        important_vals["Distance Traveled"] = [max(df_id["cum_dist"])]
        important_vals["Average Speed"] = [(df_id["speed"]).mean()]
        important_vals["Max Speed"] = [max(df_id["speed"])]
        important_vals["% Stationary"] = [(len(np.where(df_id["speed"] < 1)[0]) / len(df_id["speed"])) * 100]
        df_storage.append(important_vals) 
        
    combined_df = pd.concat(df_storage, axis = 0, ignore_index = True)
    return combined_df

def distances(dataframe, fps):
    df_storage = []
    
    # These lines below compute all of the different pairs of shrimp 
    diff_ids = dataframe["id"].unique()
    shrimp_combo = []
    for combo in combinations(diff_ids, 2):  # 2 for pairs, 3 for triplets, etc
        shrimp_combo.append(combo)
    
    # Loop through each pair of shrimp
    for pair in shrimp_combo:
        shrimpX = pair[0]
        shrimpY = pair[1]

        # This will subset our dataframe into one with just the two animals we care about
        dataframe2 = dataframe[(dataframe["id"] == shrimpX) | (dataframe["id"] == shrimpY)]

        for idx, ID in enumerate(np.unique(dataframe2['id'])):
                dataframe2['id'][dataframe2['id'] == ID] = idx
                #print(dataframe.shape)
        
        # Gets the distance between two shrimp for each frame
        distances = []
        for fr in np.unique(dataframe['frame']):
                tmp = dataframe2[dataframe2['frame'] == fr]
                x = tmp[tmp['id'] == 0]['pos_x'].values[0] - tmp[tmp['id'] == 1]['pos_x'].values[0]
                y = tmp[tmp['id'] == 0]['pos_y'].values[0] - tmp[tmp['id'] == 1]['pos_y'].values[0]
                distances.append(np.sqrt(x**2 + y**2))

        # Creates a timestamp column in seconds 
        timestamp = np.unique(dataframe2['frame'])/fps
        
        # Manipulates the timestamp column for easier-to-read results
        def convert_time(seconds):
            seconds = seconds % (24 * 3600)
            hour = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60     
            milliseconds = (seconds % 1) * 100
            milliseconds
            return "%d:%02d:%02d:%2d" % (hour, minutes, seconds, milliseconds)
        
        convert_time = np.vectorize(convert_time)
        timestamp = convert_time(timestamp)
        frames = np.unique(dataframe2['frame'])
        
        # Getting the list of the different pairs of shrimp
        pair_list = []
        for i in range(len(distances)):
                pair_list.append(pair)

        # Creates our dataframe of shrimp and their respective distance at individual moments
        dist_df = pd.DataFrame([pair_list, frames, timestamp, distances]).transpose()
        dist_df.columns = ["Pair", "Frame", "Timestamp", "Distance"]
        
        df_storage.append(dist_df)
    
    combined_df = pd.concat(df_storage, axis = 0, ignore_index = True)

    return combined_df

def timestamp(dataframe, cutoff):
    df_storage = []
    
    for pair in dataframe["Pair"].unique():
        pair_df = dataframe[dataframe["Pair"] == pair]
        
        # Creates a "Previous" column which defines where the shrimp were in the previous frame
        pair_df["Previous"] = (pair_df["Distance"].shift(1)).fillna(pair_df["Distance"])
        enter = []
        exit = []
        pair = pair_df["Pair"].iloc[0]

        # Case 2: The two shrimp are never within the distance threshold
        if (pair_df.loc[pair_df["Distance"] <= cutoff]).empty == True:
            enter.append(0)
            exit.append(0)
            label = [pair]

        # Case 2: The two shrimp are always within the distance threshold    
        elif (len(pair_df.loc[pair_df["Distance"] <= cutoff]) == len(pair_df)) == True:
            enter.append((pair_df["Timestamp"].iloc[0]))
            exit.append((pair_df["Timestamp"].iloc[-1]))
            label = [pair]

        # Case 3: The two shrimp move between being close and not close (most often) 
        else:
            if ((pair_df["Distance"].iloc[0]) <= cutoff) == True:
                enter.append((pair_df["Timestamp"].iloc[0]))
            for index, row in pair_df.iterrows():
                if ((row["Distance"] <= cutoff) & (row["Previous"] > cutoff)) == True:
                    start = row["Timestamp"]
                    enter.append(start)
                elif ((row["Distance"] > cutoff) & (row["Previous"] <= cutoff)) == True:
                    end = row["Timestamp"]
                    exit.append(end)
            if ((pair_df["Distance"].iloc[-1]) <= cutoff) == True:
                exit.append((pair_df["Timestamp"].iloc[-1]))
            label = [pair] * (len(enter))

        important_times = pd.DataFrame()
        important_times["Pair"] = label
        important_times["Start"] = enter
        important_times["End"] = exit
        df_storage.append(important_times)
    
    combined_df = pd.concat(df_storage, axis = 0, ignore_index = True)
    return combined_df


