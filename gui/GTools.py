import cv2 as cv
import numpy as np
import tkinter
from tkinter import simpledialog

# Burrow Detection
def CircleDetection(image, minD, maxR):
    '''This function takes a gray scaled image and return a list of circles detected in the image

    Parameters
    ----------
    image: ndarray, shape(n_rows, n_cols, 3)
        background image containing all three colour channels
    
        
    Returns
    -------
    circles: ndarray, shape(n_circles, 3)
        list of circles with location in x,y coordinate and a radius
    
    '''

    # Apply the Hough transform to detect circles in the image
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp = 2, minDist = minD, param1=100, param2=80, minRadius=0, maxRadius=maxR)
    # param1 and param2 need to justify as well
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    

def BurrowDetection(image, number):
    '''
    This function detects locations of burrows in the background.
    
    Parameters
    ----------
    image: ndarray, shape(n_rows, n_cols, 3)
        background image containing all three colour channels
    number: integer
        number of burrows in the background image
    
        
    Returns
    -------
    circles: ndarray, shape(n_circles, 3)
        list of circles with location in x,y coordinate and a radius
    '''
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    
    # Initial Guess of kernel size
#     x = 3
#     y = 3
#     grayBlur = cv.GaussianBlur(gray, (x, y), 0)
    
    kernel = np.ones((5, 5), np.uint8) 
    img_erosion = cv.erode(gray, kernel, iterations=5) 
    img_dilation = cv.dilate(img_erosion, kernel, iterations=5)
    ret,thresh = cv.threshold(image,70,255,0)
    # threshhold parameters need to justify
    maxR = int(min(gray.shape[0], gray.shape[1])/4)
    ini_min = 150
#     circles = CircleDetection(grayBlur)
    circles = CircleDetection(gray, minD = ini_min, maxR = maxR)
    if circles is None:
        size = 0
    else:
        size = len(circles)

    while size > number:
#         x += 2
#         y += 2
#         grayBlur = cv.GaussianBlur(gray, (x, y), 0)
        ini_min += 50 
#         circles = CircleDetection(grayBlur)
        circles = CircleDetection(gray, minD = ini_min, maxR = maxR)
        if circles is None:
            size = 0
        else:
            size = len(circles)

    if size == number:
        return circles

            
    if size != number:
        return "Needs Human Interaction"

def RegionSelection(img):
    """
    This function enable user to select 4 points on an image using mouse clicks and calculate the smallest rectangle with all 4 points included.

    Parameters:
    image: ndarray
        Input image on which points are to be selected.

    Returns:
    Points: tuple
        A tuple of 4 integers 
    """
    points = []  # List to store points
    image = np.copy(img)
    # Mouse callback function
    def VertexSelector(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                cv.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot at the point
                points.append((x, y))  # Add the point to the list
                cv.putText(image, str((x, y)), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Create a black image, a window and bind the function to window
    cv.namedWindow('image')
    cv.setMouseCallback('image', VertexSelector)

    while True:
        cv.imshow('image', image)
        key = cv.waitKey(1) & 0xFF

        if key == ord('y') and len(points) == 4:
            points.sort(key=lambda x: (x[1], x[0]))
            y = points[0][1]
            w = points[-1][1]
            points.sort(key=lambda x: (x[0], x[1]))
            x = points[0][0]
            h = points[-1][0]
            cv.destroyAllWindows()
            return (y, w, x, h)
        elif key == ord('n'):
            points = []  # Clear the points
            image = np.copy(img)  # Reset the image
        elif key == ord('q'):  # Quit the program
            cv.destroyAllWindows()
            return


def InfoProcessor(background):
    
    CropData = RegionSelection(background)
    image = np.copy(background[CropData[0]:CropData[1], CropData[2]:CropData[3]])

    number = simpledialog.askinteger("Burrow Count","How many burrows are in the arena? (Enter an integer):")
    Circles = BurrowDetection(image, number)
    centers = []
    radius = []
    for i in range(len(Circles)):
        centers.append((Circles[i][0], Circles[i][1]))
        radius.append(Circles[i][2])

    for i in range(number):
        cv.circle(image, centers[i], radius[i], (250, 100*i, 10*i), 5)
        cv.putText(image, str("Burrow {0}".format(i)), centers[i], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    while True:
        cv.imshow('image', image)
        key = cv.waitKey(1) & 0xFF

        if key == ord('y'):
            cv.destroyAllWindows()
            x = simpledialog.askinteger("Resident Location","Which burrow has a resident? (Enter an integer):")
            return CropData, centers, radius, centers[x]
        elif key == ord('n'):
            return InfoProcessor(background)
        elif key == ord('q'):  # Quit the program
            cv.destroyAllWindows()
            return

def Video2Array(video, shape = None):
    cap = cv.VideoCapture(video)

    frames = []
    ret, frame = cap.read()

    while ret:
        if shape != None:
            frame = frame[shape[0]:shape[1], shape[2]:shape[3]]
        frames.append(frame)
        ret, frame = cap.read()
    return frames    

def VideoClip(Video, output, start = 0, end = None, mode = 0):
    '''
    This function crops the video to certain shape.
    
    Parameters
    ----------
    Video: string
        file path of the video to be implemented
    shape: a tuple of 4 integers
        the edge points of the cropped video
        
    Returns
    -------
    processed: ndarray
        a cropped video clip
    '''
    cap = cv.VideoCapture(Video)
    cap.set(cv.CAP_PROP_POS_MSEC, start)
    ret, frame = cap.read()

    size = (frame.shape[1], frame.shape[0])
    
    processed = []
    if mode == 1:
        out = cv.VideoWriter(output, cv.VideoWriter_fourcc(*'MJPG'), 29, size)
        while ret:
            
            if end != None:
                if not ret or cap.get(cv.CAP_PROP_POS_MSEC) >= end:
                    break
            processed.append(frame)
            out.write(frame)

            ret, frame = cap.read()
        out.release()
    else:
        while ret:
            
            if end != None:
                if not ret or cap.get(cv.CAP_PROP_POS_MSEC) >= end:
                    break
            processed.append(frame)

            ret, frame = cap.read()
    cap.release()

    return processed

def BGExtraction(Video, frame, option=1):
    '''
    This function extracts the background of the video.
    
    Parameters
    ----------
    Video: string
        file path of the video to be implemented
    frame: integer
        the frame of the video to be served as the background
    option: integer, 0 or 1
        default value 1 and export the background image if set to 0
        
    Returns
    -------
    bg: ndarray
        selected background image of the video
    '''
    cap = cv.VideoCapture(Video)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)
    ret, bg = cap.read()

    if option == 0:
        cv.imwrite("bg.jpg", bg)

    cap.release()

    return bg

def SingleImageProcessor(frame, threshold):
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)) 
    opened = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, iterations = 2)
    blur = cv.GaussianBlur(opened, (55, 55), 0) 
    threshold = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY_INV)[1]

    return threshold