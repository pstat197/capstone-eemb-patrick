import cv2 as cv
import numpy as np

# Burrow Detection
def CircleDetection(image):
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
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 20, param1=60, param2=40, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    
def OddGenerator(number):
    
    odd = []
    i = 1
    while i <= number:
        odd.append(i)
        i += 2

    return odd

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
    x = 7
    y = 7
    grayBlur = cv.GaussianBlur(gray, (x, y), 0)

    circles = CircleDetection(grayBlur)
    if circles is None:
        size = 0
    else:
        size = len(circles)

    # Case I: number of circles detected >  number
    while size > number:
        x += 2
        y += 2
        grayBlur = cv.GaussianBlur(gray, (x, y), 0)

        circles = CircleDetection(grayBlur)
        if circles is None:
            size = 0
        else:
            size = len(circles)

    if size == number:
        return circles

    # Case II: number of circles detected < number
    x_odd = OddGenerator(x)
    y_odd = OddGenerator(y)

    for i in x_odd:
        for j in y_odd:
            grayBlur = cv.GaussianBlur(gray, (i, j), 0)
            circles = CircleDetection(grayBlur)
            if circles is None:
                size = 0
            else:
                size = len(circles)
            
            if size == number:
                return circles
            
    if size != number:
        return "Not enough circles detected."


# Background Subtraction
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

def VideoCrop(Video, shape):
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
    gvideo: ndarray
        a grayscaled cropped version of the video
    '''

    cap = cv.VideoCapture(Video)
    ret, frame = cap.read()

    size = (shape[3]-shape[2], shape[1]-shape[0])

    out = cv.VideoWriter("Cropped.avi", cv.VideoWriter_fourcc(*'MJPG'), 29, size)

    while ret:
        Crop = frame[shape[0]:shape[1], shape[2]:shape[3]]
        Crop = cv.cvtColor(Crop, cv.COLOR_RGB2GRAY)

        RGBMode = cv.cvtColor(Crop, cv.COLOR_GRAY2RGB)
        out.write(RGBMode)

        ret, frame = cap.read()

    out.release()
    cap.release()

def VideoClip(Video, output,  start, end):

    cap = cv.VideoCapture(Video)
    cap.set(cv.CAP_PROP_POS_MSEC, start)

    
    duration = end
    ret, frame = cap.read()
    size = (frame.shape[1], frame.shape[0])
    out = cv.VideoWriter(output, cv.VideoWriter_fourcc(*'MJPG'), 29, size)

    # Read frames from the video until the end of the clip is reached
    while ret:
        if not ret or cap.get(cv.CAP_PROP_POS_MSEC) >= duration:
            break
        out.write(frame)

        ret, frame = cap.read()

    cap.release()
    out.release()


def SubtractionThreshhold(imageTensor):
    mean = np.average(imageTensor)
    std = np.std(imageTensor)
    thresh = int(mean+2*std)
    threshold = cv.threshold(imageTensor, thresh, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    return threshold


def BackgroundSubtraction(video, back, output, start):
    '''
    This function applies background subtraction to the entire video.
    
    Parameters
    ----------
    Video: string
        file path of the video to be implemented
    back: ndarray, shape(n_rows, n_cols, 3)
        background image containing all three colour channels
    start: integer
        the starting point of the subtraction, in milliseconds
        
    Returns
    -------
    gvideo: ndarray
        a grayscaled cropped version of the video
    '''
    if isinstance(back, str):
        background = cv.imread(back)
    else:
        background = back
    capture = cv.VideoCapture(video)
    capture.set(cv.CAP_PROP_POS_MSEC, start)

    ret, frame = capture.read()

    size = (frame.shape[1], frame.shape[0])

    # Convert background to required format
    background = cv.resize(background, size)
    #grayBackground = cv.GaussianBlur(background, (55,55),0)
    grayBackground = cv.cvtColor(background, cv.COLOR_RGB2GRAY)
    # Gaussian blur
    
    processed = []
    out = cv.VideoWriter(output, cv.VideoWriter_fourcc(*'MJPG'), 29, size)

    while ret:

        # Convert frame to required format
        #grayFrame = cv.GaussianBlur(frame, (55,55),0)
        grayFrame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # Gaussian blur

        # Subtract the background
        delta = cv.absdiff(grayBackground, grayFrame)

        # removes small struture

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)) 
        opened = cv.morphologyEx(delta, cv.MORPH_OPEN, kernel)

        #kernel = GaussianBlurKernel(opened)
        blur = cv.GaussianBlur(opened, (55, 55), 0) 

        # fills the gap
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
        closed = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)
        
        threshold = cv.threshold(closed, 165, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
        

        #Record the processed image
        processed.append(threshold)

        RGBMode = cv.cvtColor(threshold, cv.COLOR_GRAY2RGB)
        out.write(RGBMode)

        ret, frame = capture.read()

    capture.release()
    out.release()

    return processed