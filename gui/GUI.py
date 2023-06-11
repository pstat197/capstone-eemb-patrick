#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import GTools as gt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import cv2 as cv
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from itertools import combinations
pd.options.mode.chained_assignment = None
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
from PIL import ImageTk, Image
import cv2 as cv
import tracktor as tr
import os
import multiprocessing as mp
from scipy.spatial import distance
from itertools import combinations
from tkinter import simpledialog



root = tk.Tk()
root.title("EEMB DS Capstone")
root.geometry("800x800")

tabControl = ttk.Notebook(root)

tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)

img = Image.open("/Users/ashleyson/Desktop/gui/mantis-shrimp-green.png")
my_img = img.resize((500, 380))
orig_img = ImageTk.PhotoImage(my_img)
ttk.Label(tab1, image=orig_img).grid(column=0, row=0, sticky="n")

tabControl.add(tab1, text="Home")
tabControl.add(tab2, text ="Tracking")
tabControl.add(tab3, text="Obtain Key Stats + Timestamp")
tabControl.pack(expand=1, fill="both")

tab1.rowconfigure(0, minsize=50, weight=1)
tab1.columnconfigure([0, 1, 2], minsize=50, weight=1)

###############################

def getBackground(video, n):
    video = cv.VideoCapture(video)
    # count the total frames in the video 
    count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    # calculate the modulo
    modulo = count % n
    # select equally spaced frames across the video 
    frame_index = np.linspace(0, count - modulo, n+1).astype(np.int64)
    # set up lists to hold each color channel
    frames_b = []
    frames_g = []
    frames_r = []
    counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if np.isin(counter, frame_index):
                # get the frame and put each color channel to the list
                frames_b.append(frame[:,:,0].astype(np.int64))
                frames_g.append(frame[:,:,1].astype(np.int64))
                frames_r.append(frame[:,:,2].astype(np.int64))
            counter += 1
        else: 
            break
    # stack all frames together and take the median
    stacked_b = np.stack(frames_b, axis = 2)
    median_b = np.abs(np.median(stacked_b, axis = 2))
    stacked_g = np.stack(frames_g, axis = 2)
    median_g = np.abs(np.median(stacked_g, axis = 2))
    stacked_r = np.stack(frames_r, axis = 2)
    median_r = np.abs(np.median(stacked_r, axis = 2))
    # merge median frame together to one frame
    background = cv.merge((median_b, median_g, median_r))
    video.release()
    background = cv.cvtColor(background.astype(np.uint8), cv.COLOR_BGR2RGB)
    return background

##########################################################       

def MasterTracking(video_path, background_path = 0):

    # Read in background images
    if background_path == 0:
        background = getBackground(video_path, 100)
    elif isinstance(background_path, str):
        background = cv.imread(background_path)

    # Collect user inputs
    shape, centers, radius, initBurrow = gt.InfoProcessor(background)
    n_inds = simpledialog.askinteger("Object Count","How many objects do you want to track? (Enter an integer):")
    t_id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    colours = [(0,0,255),(0,255,255),(255,0,255),(255,255,255),(255,255,0),(255,0,0),(0,255,0),(0,0,0)]
    meas_last = list(np.zeros((2,2)))
    meas_now = list(np.zeros((2,2)))
    df = []
    last = 0

    # Basic conversion and data collections of the background and croppedBackground
    size = (background.shape[1], background.shape[0])
    background = cv.cvtColor(background, cv.COLOR_RGB2GRAY)

    # Start video processing
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    while ret:
        this = cap.get(1)
        print(this)
        frame = cv.resize(frame, size)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        delta = cv.absdiff(background, frame)
        delta = delta[shape[0]:shape[1], shape[2]:shape[3]]

        # Starts tracking
        thresh = gt.SingleImageProcessor(delta)
        final, contours, meas_last, meas_now = tr.detect_and_draw_contours(frame, thresh, meas_last, meas_now, 100, 10000)

        # Clear duplicated contours
        value = []
        for x,y in combinations(meas_now, 2):
            if distance.euclidean(x, y) < 8 and not (y in value):
                value.append(y)
        for i in value:
            meas_now.remove(i)

        while len(meas_now) < len(meas_last):
            if this == 1:
                meas_now.append(initBurrow)
                if len(meas_now) == len(meas_last):
                    break
            meas_now.append([0,0])
        row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)

        # reorder
        equal = np.array_equal(col_ind, list(range(len(col_ind))))
        if equal == False:
            current_ids = col_ind.copy()
            reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
            meas_now = [x for (y,x) in sorted(zip(reordered,meas_now))]
            
        if len(meas_now) > n_inds:
            meas_now = meas_now[0:n_inds]

        for i, meas in enumerate(meas_now):
            if meas == [0,0]:
                meas_now[i] = meas_last[i]

        for i in range(n_inds):
            df.append([this, meas_now[i][0]+shape[2], meas_now[i][1]+shape[0], t_id[i]])

        #cv.imshow("thresh", thresh)
        #if cv.waitKey(1) & 0xFF == ord('q'):
        #    break
        #Record the processed image


        if last >= this:
            break
        
        last = this
        ret, frame = cap.read()

    cv.destroyAllWindows()
    cap.release()

    df = pd.DataFrame(np.matrix(df), columns = ['frame','pos_x','pos_y', 'id'])
    return df

##########################################################

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
    video_entry.delete(0, tk.END)
    video_entry.insert(0, file_path)


def select_background():
    file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
    background_entry.delete(0, tk.END)
    background_entry.insert(0, file_path)
        

    
def run_tracking():
    video_path = video_entry.get()
    background_path = background_entry.get()
    df = MasterTracking(video_path, background_path)
    
    df.to_csv("output.csv", index = False)
###########################################################

# Create a label and entry field for video input
video_label = ttk.Label(tab2, text="Video:")
video_label.pack()
video_entry = ttk.Entry(tab2)
video_entry.pack()

# Create a button to select the video file
video_button = ttk.Button(tab2, text="Select Video", command=select_video)
video_button.pack()

# Create a label and entry field for background input
background_label = ttk.Label(tab2, text="Background:")
background_label.pack()
background_entry = ttk.Entry(tab2)
background_entry.pack()

# Create a button to select the background file
background_button = ttk.Button(tab2, text="Select Background", command=select_background)
background_button.pack()

# Create a button to run the tracking function
run_button = ttk.Button(tab2, text="Run Tracking", command=run_tracking)
run_button.pack()



########### TAB 3: EVERYTHING ##########


def everything():
    dataframe = df_everything
    name = file_name
    path = path_everything
    amt_of_shrimp = shrimp_count
    cutoff = thresh
    fps = 29.777777777
    '''Key Values Below'''
    df_storage_kv = []
    df_storage_dist = []
    df_storage_ts = []


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
        df_storage_kv.append(important_vals) 
        
    combined_df_kv = pd.concat(df_storage_kv, axis = 0, ignore_index = True)
    
    
    '''Distances Below'''
    diff_ids = dataframe["id"].unique()
    shrimp_combo = []
    for combo in combinations(diff_ids, amt_of_shrimp):  # 2 for pairs, 3 for triplets, etc
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
        
        df_storage_dist.append(dist_df)
    
    combined_df_dist = pd.concat(df_storage_dist, axis = 0, ignore_index = True)
    
    '''Timestamps Below'''
    for pair in combined_df_dist["Pair"].unique():
        pair_df = combined_df_dist[combined_df_dist["Pair"] == pair]
        
        # Creates a "Previous" column which defines where the shrimp were in the previous frame
        pair_df["Previous"] = (pair_df["Distance"].shift(1)).fillna(pair_df["Distance"])
        enter = []
        exit = []
        pair = pair_df["Pair"].iloc[0]

        # Case 2: The two shrimp are never within the distance threshold
        if pair_df.loc[(pair_df["Distance"] <= cutoff) & (pair_df["Distance"] > 5)].empty == True:
            enter.append(0)
            exit.append(0)
            label = [pair]

        # Case 2: The two shrimp are always within the distance threshold    
        elif (len(pair_df.loc[(pair_df["Distance"] <= cutoff) & (pair_df["Distance"] > 5)]) == len(pair_df)) == True:
            enter.append((pair_df["Timestamp"].iloc[0]))
            exit.append((pair_df["Timestamp"].iloc[-1]))
            label = [pair]

        # Case 3: The two shrimp move between being close and not close (most often) 
        else:
            if ((pair_df["Distance"].iloc[0]) <= cutoff) == True:
                enter.append((pair_df["Timestamp"].iloc[0]))
            for index, row in pair_df.iterrows():
                if ((row["Distance"] > 5) & (row["Distance"] <= cutoff) & (row["Previous"] > cutoff)) == True:
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
        df_storage_ts.append(important_times)
    
    combined_df_ts = pd.concat(df_storage_ts, axis = 0, ignore_index = True)
    
    kv_output = path + "/" + str(name) + "_keyvalues.csv"
    combined_df_kv.to_csv(kv_output, sep = ",")

    dist_output = path + "/" + str(name) + "_distances.csv"
    combined_df_dist.to_csv(dist_output, sep = ",")

    ts_output = path + "/" + str(name) + "_timestamps.csv"
    combined_df_ts.to_csv(ts_output, sep = ",")
    
def upload_df_tab3():
    global df_everything
    file_types = [("CSV files", "*.csv"), ("All", "*.*")]
    file = filedialog.askopenfilename(filetypes = file_types)
    df_everything = pd.read_csv(file)
    return df_everything

def get_name():
    global file_name
    file_name = file_name_input_tab3.get()
    file_name = str(file_name)
    return file_name

def upload_path_tab3():
    global path_everything
    file_dir = filedialog.askdirectory()
    path_everything = str(file_dir)
    return path_everything

def get_amt_of_shrimp():
    global shrimp_count
    shrimp_count = shrimp_count_input_tab3.get()
    shrimp_count = int(shrimp_count)
    return shrimp_count

def get_threshold():
    global thresh
    thresh = threshold_input_tab3.get()
    thresh = int(thresh)
    return thresh


raw_csv_text_tab3 = tk.Label(tab3, width = 40, text= "CSV File you want to track", bg = "lightblue")
raw_csv_text_tab3.grid(row = 2, column=0, sticky="e", padx=50, pady=20)
raw_csv_input_tab3 = ttk.Button(master = tab3, text= "Browse", 
                         command = lambda:upload_df_tab3())
raw_csv_input_tab3.grid(row = 2, column=1, sticky="nsew", padx=50, pady=20)


file_path_text_tab3 = tk.Label(tab3, width = 40, text= "Designated Output Path", bg = "lightblue")
file_path_text_tab3.grid(row = 3, column=0, sticky="e", padx=50, pady=20)
file_path_input_tab3 = ttk.Button(master = tab3, text= "Browse", 
                         command = lambda:upload_path_tab3())
file_path_input_tab3.grid(row = 3, column=1, sticky="nsew", padx=50, pady=20)


threshold_text_tab3 = ttk.Button(master = tab3, width = 40, text = "Input Closeness Threshold: ", 
                               command = lambda:get_threshold())
threshold_text_tab3.grid(row = 4, column = 0, pady = 20)
threshold_input_tab3 = ttk.Entry(tab3)
threshold_input_tab3.grid(row = 4, column = 1, sticky = "w", pady = 20)

file_name_text_tab3 = ttk.Button(master = tab3, width = 40, text = "Input File Identification Name: ", 
                               command = lambda:get_name())
file_name_text_tab3.grid(row = 5, column = 0, pady = 20)
file_name_input_tab3 = ttk.Entry(tab3)
file_name_input_tab3.grid(row = 5, column = 1, sticky = "w", pady = 20)

shrimp_count_text_tab3 = ttk.Button(master = tab3, width = 40, text = "Input Amount of Objects to Track: ", 
                               command = lambda:get_amt_of_shrimp())
shrimp_count_text_tab3.grid(row = 6, column = 0, pady = 20)
shrimp_count_input_tab3 = ttk.Entry(tab3)
shrimp_count_input_tab3.grid(row = 6, column = 1, sticky = "w", pady = 20)

generate_button_tab3= Button(master = tab3, text = "Generate Important Timestamps from this Dataset", width = 30,
                             command = lambda: everything())
generate_button_tab3.grid(row = 7, column = 0, pady = 40, sticky = "e")


    
        
    
ttk.Label(tab3, text = "View above").grid(row = 8, column = 0)
    
    
##################
img = Image.open("/Users/ashleyson/Desktop/gui/mantis-shrimp-green.png")
my_img = img.resize((500, 380))
orig_img = ImageTk.PhotoImage(my_img)
ttk.Label(tab1, image=orig_img).grid(column=0, row=0, sticky="n") 

ttk.Label(tab1, text="Welcome to the EEMB Data Science Capstone GUI!").grid(column=0, row=1)

ttk.Label(tab1, text="Navigate throughout the tabs to obtain key information about the shrimp interactions!").grid(column=0, row=2, pady=20)

ttk.Label(tab1, text="Movement Statistics: Important info like shrimp speed, distance traveled, etc.").grid(column=0, row=3)

ttk.Label(tab1, text="Distances Between Shrimp: The distance between each pair of shrimp across each frame").grid(column=0, row=4)

ttk.Label(tab1, text="Important Timestamps: Ranges of time when interactions are occuring between shrimp").grid(column=0, row=5)

root.mainloop()


# In[ ]:




