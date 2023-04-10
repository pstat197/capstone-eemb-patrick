import cv2 as cv
import numpy as np

# Burrow Detection
def BurrowDetection(background):
    '''This function takes a gray scaled image and return a list of circles detected in the image'''

    # Apply the Hough transform to detect circles in the image
    circles = cv.HoughCircles(background, cv.HOUGH_GRADIENT, 1, 20, param1=60, param2=40, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    else:
        print("No circle is detected.")




# Background Subtraction

def SubtractionThreshhold(imageTensor):
    mean = np.average(imageTensor)
    std = np.std(imageTensor)
    thresh = int(mean+2*std)
    threshold = cv.threshold(imageTensor, thresh, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    return threshold


def BackgroundSubtraction(video, back):
    background = cv.imread(back)
    capture = cv.VideoCapture(video)

    ret, frame = capture.read()

    size = (frame.shape[1], frame.shape[0])

    # Convert background to required format
    background = cv.resize(background, size)
    #grayBackground = cv.GaussianBlur(background, (55,55),0)
    grayBackground = cv.cvtColor(background, cv.COLOR_RGB2GRAY)
    
    processed = []
    out = cv.VideoWriter("BlackWhiteResult.avi", cv.VideoWriter_fourcc(*'MJPG'), 29, size)

    while ret:

        # Convert frame to required format
        #grayFrame = cv.GaussianBlur(frame, (55,55),0)
        grayFrame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # Subtract the background
        delta = cv.absdiff(grayBackground, grayFrame)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        opened = cv.morphologyEx(delta, cv.MORPH_OPEN, kernel)

        blur = cv.GaussianBlur(opened, (23, 23), 0)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
        closed = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)
        
        threshold = cv.threshold(closed, 135, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]

        #Record the processed image
        processed.append(threshold)

        RGBMode = cv.cvtColor(threshold, cv.COLOR_GRAY2RGB)
        out.write(RGBMode)

        ret, frame = capture.read()

    capture.release()
    out.release()

    return processed