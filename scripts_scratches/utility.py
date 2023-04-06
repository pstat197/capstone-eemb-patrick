import cv2 as cv
import numpy as np

# Burrow Detection
def BurrowDetection(background):

    gray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (105, 105), 0)

    # Apply the Hough transform to detect circles in the image
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    # If circles were found, draw a circle around each one and assign a unique ID
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    else:
        print("No circle is detected.")

def BurrowInteractiveAreaGraph(image, circles):
        
    for i, (x, y, r) in enumerate(circles):

        cv.circle(image, (x, y), 1.5*r, (10*i, 0, 0), 2)
        cv.circle(image, (x, y), r, (0, 10*i, 0), 2)
        cv.putText(image, "C"+i, (x - r, y - r), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image







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
    grayBackground = cv.cvtColor(background, cv.COLOR_RGB2GRAY)
    grayBackgroundBlur = cv.GaussianBlur(grayBackground, (105,105),0)

    #processed = []
    out = cv.VideoWriter("BlackWhiteResult.avi", cv.VideoWriter_fourcc(*'MJPG'), 29, size)

    while ret:

        # Convert frame to required format
        grayFrame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        grayFrameBlur = cv.GaussianBlur(grayFrame, (105,105),0)

        # Subtract the background
        delta = cv.absdiff(grayBackgroundBlur, grayFrameBlur)
        threshold = cv.threshold(delta, 135, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]

        # Record the processed image
        #processed.append(threshold)
        RGBMode = cv.cvtColor(threshold, cv.COLOR_GRAY2RGB)
        out.write(RGBMode)

        ret, frame = capture.read()

    capture.release()
    out.release()