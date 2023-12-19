from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import depthai as dai


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

pts_lst = []


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

colorCamera = pipeline.create(dai.node.ColorCamera)
colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCamera.setIspScale(1,3)
# colorCamera.setVideoSize(1920,1080)
# colorCamera.initialControl.setSharpness(0)
# colorCamera.setPreviewSize(1920,1080)

xoutRGB = pipeline.create(dai.node.XLinkOut)
xoutRGB.setStreamName("rgb")
xoutRGB.input.setBlocking(False)
xoutRGB.input.setQueueSize(1)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
colorCamera.isp.link(xoutRGB.input)
# colorCamera.preview.link(xoutRGB.input)


xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)



lrcheck = True
subpixel = True


stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)


#Config

topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)


spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)



with dai.Device(pipeline) as device:

      # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    colorQueue = device.getOutputQueue(name = 'rgb', maxSize = 1, blocking = False)


    color = (100,100,0)
    # time.sleep(2.0)

    while True:
        
        #get current frame from videostream
        # frame = vs.read()
        frame = colorQueue.get().getCvFrame()

        frame = cv2.flip(frame,flipCode = 1)

        frame = frame[1] if args.get("video",False) else frame

        if frame is None:
            break

         #frame modifications
        # frame = imutils.resize(frame, width = 600)
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # frame = cv2.resize(frame,(640,400))

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

            if radius > 5:
                #circle for ball
                cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
                #circle for centroid
                cv2.circle(frame,center,5,(0,0,255),-1)
                font = cv2.FONT_HERSHEY_COMPLEX
                # cv2.putText(frame,str(cv2.contourArea(c)),center,font,0.5,(0,255,0))
                cv2.putText(frame,str(center),center,font,0.7,(0,0,0))

        pts.appendleft(center)



        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue   
            
            thickness = int(np.sqrt(args["buffer"]/float(i+1)) * 2.5)
            cv2.line(frame, pts[i-1],pts[i],(0,255,0),thickness)

        frame_dim = frame.shape
        # print(frame_dim[0])
        # print("(Hellooo,{}".format(frame_dim))


        if center is not None and radius is not None:

            topLeft = dai.Point2f((int(center[0])-int(radius))/640,(int(center[1])-int(radius))/360)
            bottomRight = dai.Point2f((int(center[0])+int(radius))/640,(int(center[1])+int(radius))/360)

            # topLeft = dai.Point2f(0.2,0.2)
            # bottomRight = dai.Point2f(0.8,0.8)
            # print("FIRST IF")
            # print(frame_dim)

        else:
            topLeft = dai.Point2f(0.4,0.4)
            bottomRight = dai.Point2f(0.6,0.6)
            # print("SECOND IF")
        
        config.roi = dai.Rect(topLeft, bottomRight)
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        spatialCalcConfigInQueue.send(cfg)


        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        depthFrame = inDepth.getFrame() # depthFrame values are in millimeters
        depthFrame = cv2.flip(depthFrame,flipCode = 1)

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        # combinedFrame = np.array(frame/2 + depthFrameColor/2)
        # cv2.imshow("Combined",combinedFrame)
        

        spatialData = spatialCalcQueue.get().getSpatialLocations()

        for depthData in spatialData:

            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            x_coord = int(depthData.spatialCoordinates.x)
            y_coord = int(depthData.spatialCoordinates.y)
            z_coord = int(depthData.spatialCoordinates.z)


            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, (255,255,255))
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, (255,255,255))
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, (255,255,255))

            pts_lst.append((x_coord,y_coord,z_coord))

        # Show the frame
        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("contour",mask)
        cv2.imshow("Color Camera", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if z_coord < 2000 :
            print('Too close!!!')

        else:
            print("Okie Dokie!")
            

cv2.destroyAllWindows()