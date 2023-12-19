import cv2 as cv
import numpy as np
 # import freenect
 # def get_video():
 #     array,_ = freenect.sync_get_video()
 #     array = cv.cvtColor(array,cv.COLOR_RGB2BGR)
 #     return array
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == True:
        # red = np.uint8([[[0, 0, 255]]])

        # Converting into hsv color space
        red_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # print(red_hsv)
        # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # To create mask use +10 and -10 of the hue value for upper and lower limits 
        hsv_lb = np.array([0, 180, 100])
        hsv_ub = np.array([10, 255, 255])
        thresh1 = cv.inRange(red_hsv, hsv_lb, hsv_ub)


        hsv_lb = np.array([170, 180, 100])
        hsv_ub = np.array([180, 255, 255])
        thresh2 = cv.inRange(red_hsv, hsv_lb, hsv_ub)

        thresh = thresh1 | thresh2
        cv.imshow('mask', thresh)

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # green = frame[:, :, 1]
        # diff = cv.subtract(green, gray)
        # _, thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

        thresh = cv.medianBlur(thresh, 5)
        thresh = cv.Canny(thresh, 100, 200)
        cv.imshow('edge', thresh)
        cv.waitKey(1)

        rows = thresh.shape[0]
        circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, 1, rows / 8,
                                param1=30, param2=50,
                                minRadius=10, maxRadius=100)

    # cv.imshow("binary", thresh)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            # cv.circle(src, center, 1, (0, 100, 100), 5)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (0, 0, 0), 5)
            print(center)


    # cv.imshow("detected circles", frame)
    # cv.waitKey(1)
    # cv.destroyAllWindows()