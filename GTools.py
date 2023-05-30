import cv2 as cv
import numpy as np
import BGTools as bgt

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


def InfoProcessor(back):

    if isinstance(back, str):
        background = cv.imread(back)
    else:
        background = back
    
    CropData = RegionSelection(background)
    background = background[CropData[0]:CropData[1], CropData[2]:CropData[3]]

    number = int(input("How many burrows are in the arena? (Enter an integer):"))
    Circles = bgt.BurrowDetection(background, number)
    centers = []
    radius = []
    for i in range(len(Circles)):
        centers.append((Circles[i][0], Circles[i][1]))
        radius.append(Circles[i][2])

    image = np.copy(background)
    for i in range(number):
        cv.circle(image, centers[i], radius[i], (250, 100*i, 10*i), 5)
        cv.putText(image, str("Burrow {0}".format(i)), centers[i], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    while True:
        cv.imshow('image', image)
        key = cv.waitKey(1) & 0xFF

        if key == ord('y'):
            cv.destroyAllWindows()
            x = int(input("Which burrow has a resident? (Enter an integer):"))
            return background, CropData, centers, radius,centers[x]
        elif key == ord('n'):
            return InfoProcessor(back)
        elif key == ord('q'):  # Quit the program
            cv.destroyAllWindows()
            return
        
def BGSelector(Video):

    n = int(input("How many frames do you want to use to generate the background? (Enter an integer):"))
    background = bgt.getBackground(Video, n)

    while True:
        cv.imshow('image', background)
        key = cv.waitKey(1) & 0xFF

        if key == ord('y'):
            cv.destroyAllWindows()
            return background
        elif key == ord('n'):
            return BGSelector(Video)
        elif key == ord('r'):  # Quit the program
            file = input("Please enter the file path of your own background:")
            cv.destroyAllWindows()
            return file
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