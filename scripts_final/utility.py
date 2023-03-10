import cv2 as cv
import numpy as np

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