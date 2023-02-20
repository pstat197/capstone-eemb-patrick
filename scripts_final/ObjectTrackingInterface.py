import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import utility

video = input("Please include the local path to the video: ")
background = input("Please include the local path to the background of the above video: ")

utility.BackgroundSubtraction(video, background)