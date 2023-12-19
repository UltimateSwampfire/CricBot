from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import depthai as dai

#Define pipeline

pipeline = dai.Pipeline()

#Creating Nodes
colorCamera = pipeline.create(dai.node.ColorCamera)
colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)

xoutRGB = pipeline.create(dai.node.XLinkOut)
xoutRGB.setStreamName("rgb")

#properties

# colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
# colorCamera.setPreviewSize(300,300)

#Linking
colorCamera.preview.link(xoutRGB.input) #Currently uses preview cuz it is the most responsive, need a bigger view later


#Constructing arguement parses

ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help = "path to the (optional) video file")
ap.add_argument("-b","--buffer", type = int, default = 16, help = "max buffer size")
ap.add_argument("-u","--hsv", type = str, default = 'g', help = "Ball color")

args = vars(ap.parse_args())

print(args)
#Define lower and upper boundaries of the ball color


if args["hsv"] == "g": #green
    colorLower = (27, 90, 0)
    colorUpper = (38, 255, 255)
elif args["hsv"] == "y": #yellow

    colorLower = (22, 132, 72)
    colorUpper = (28, 255, 207)
elif args["hsv"] == "r": #red
    colorLower = (0, 106, 0)
    colorUpper = (6, 177, 255)

# #Green
# colorLower = (30, 54, 0)
# colorUpper = (52, 209, 255)

pts = deque(maxlen = args["buffer"])

#if --video not provided, access the webcam

# if not args.get("video", False):  #aka if there is no video file given
#     vs = VideoStream(src = 0).start()  #set webcam as your videostream

# #if video was infact provided, capture it
# else:
#     vs = cv2.VideoCapture(args["video"])



with dai.Device(pipeline) as device:

    colorQueue = device.getOutputQueue(name = 'rgb')

    #Allow camera to 'warm-up'?
    time.sleep(2.0)
    #keep looping
    while True:
        #get current frame from videostream
        # frame = vs.read()
        frame = colorQueue.get().getCvFrame()

        frame = cv2.flip(frame,flipCode = 1)

        frame = frame[1] if args.get("video",False) else frame

        if frame is None:
            break

        #frame modifications
        frame = imutils.resize(frame, width = 600)
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask

        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)

        #find contours and initialize (x,y) coordso of ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            #find largest contour in the mask
            #Use it to compute minimum enclosing circle and centroid
            c = max(cnts, key = cv2.contourArea)
            ((x,y),radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            # center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            center = (int(x),int(y))
            #Ball size threshold
            if radius > 5:
                #circle for ball
                cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
                #circle for centroid
                cv2.circle(frame,center,5,(0,0,255),-1)
                font = cv2.FONT_HERSHEY_COMPLEX
                # cv2.putText(frame,str(cv2.contourArea(c)),center,font,0.5,(0,255,0))
                cv2.putText(frame,str(center),center,font,0.7,(0,0,0))

        #update points queue
        pts.appendleft(center)

        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            
            
            thickness = int(np.sqrt(args["buffer"]/float(i+1)) * 2.5)
            cv2.line(frame, pts[i-1],pts[i],(0,255,0),thickness)

        
        cv2.imshow("Frame",frame)
        # cv2.imshow("Contour",mask)
        print(str(center))
        key = cv2.waitKey(1) & 0xFF


        #quit if the key 'q' is pressed
        if key == ord("q"):
            break

    # if not args.get("video",False):
    #     vs.stop()

    # else:
    #     vs.release()

cv2.destroyAllWindows()







