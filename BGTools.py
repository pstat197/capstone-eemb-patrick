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
    x = 3
    y = 3
    grayBlur = cv.GaussianBlur(gray, (x, y), 0)

    circles = CircleDetection(grayBlur)
    if circles is None:
        size = 0
    else:
        size = len(circles)

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

            
    if size != number:
        return "Needs Human Interaction"


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

def getBackground(Video, n):
    video = cv.VideoCapture(Video)
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
    return background


def VideoClipCrop(Video, output, shape = None, start = 0, end = None, mode = 0):
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

    if shape != None:
        size = (shape[3]-shape[2], shape[1]-shape[0])
    else:
        size = (frame.shape[1], frame.shape[0])
    
    processed = []
    if mode == 1:
        out = cv.VideoWriter(output, cv.VideoWriter_fourcc(*'MJPG'), 29, size)
        while ret:
            frame = frame[shape[0]:shape[1], shape[2]:shape[3]]
            
            if end != None:
                if not ret or cap.get(cv.CAP_PROP_POS_MSEC) >= end:
                    break
            processed.append(frame)
            out.write(frame)

            ret, frame = cap.read()
        out.release()
    else:
        while ret:
            frame = frame[shape[0]:shape[1], shape[2]:shape[3]]
            
            if end != None:
                if not ret or cap.get(cv.CAP_PROP_POS_MSEC) >= end:
                    break
            processed.append(frame)

            ret, frame = cap.read()
    cap.release()

    return processed

    
    


def SubtractionThreshhold(imageTensor):
    mean = np.average(imageTensor)
    std = np.std(imageTensor)
    thresh = int(mean+2*std)
    threshold = cv.threshold(imageTensor, thresh, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    return threshold



def BackgroundSubtraction(video, back, shape = None, start=0, end = None):
    '''
    This function applies background subtraction to the entire video.
    
    Parameters
    ----------
    Video: string
        file path of the video to be implemented
    back: ndarray, shape(n_rows, n_cols, 3)
        background image containing all three colour channels
    shape: tuple
        a tuple of 4 integers specifying the dimension of cropped image
    start: integer
        the starting point of the subtraction, in milliseconds
    end: integer, default None
        the end point of the subtraction, in milliseconds; If None, then the end point is the end of video
        
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

    grayBackground = cv.cvtColor(background, cv.COLOR_RGB2GRAY)
    # Gaussian blur
    
    processed = []

    while ret:

        if end != None:
            if capture.get(cv.CAP_PROP_POS_MSEC) >= end:
                break

        if shape != None:
            frame = frame[shape[0]:shape[1], shape[2]:shape[3]]

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
        blur = cv.GaussianBlur(opened, (33, 33), 0) 
        blur = cv.GaussianBlur(blur, (55, 55), 0) 

        # fills the gap
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
        #closed = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)
        
        equalized = cv.normalize(blur, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

        threshold = cv.threshold(equalized, 205, 255, cv.THRESH_BINARY_INV)[1]
        

        cv.imshow("threshold", threshold)

        # Exit if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        #Record the processed image
        processed.append(threshold)

        ret, frame = capture.read()
    cv.destroyAllWindows()
    capture.release()

    return processed

def BackgroundSubtraction2(video, back):
    '''
    This function applies background subtraction to the entire video.
    
    Parameters
    ----------
    Video: string
        file path of the video to be implemented
    back: ndarray, shape(n_rows, n_cols, 3)
        background image containing all three colour channels
        
    Returns
    -------
    processed: ndarray
        a grayscaled cropped version of the video
    '''
    if isinstance(back, str):
        background = cv.imread(back)
    else:
        background = back

    grayBackground = cv.cvtColor(background, cv.COLOR_RGB2GRAY)
    # Gaussian blur
    
    processed = []
    
    for i in range(len(video)):

        frame = video[i]
        grayFrame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        delta = cv.absdiff(grayBackground, grayFrame)


        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)) 
        opened = cv.morphologyEx(delta, cv.MORPH_OPEN, kernel)

        blur = cv.GaussianBlur(opened, (33, 33), 0) 
        blur = cv.GaussianBlur(blur, (55, 55), 0) 

        
        equalized = cv.normalize(blur, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

        threshold = cv.threshold(equalized, 205, 255, cv.THRESH_BINARY_INV)[1]
        
        processed.append(threshold)

    return processed