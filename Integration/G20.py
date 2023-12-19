import depthai as dai
import cv2
import time
import numpy as np
import imutils
from collections import deque

pts = deque(maxlen = 10) 
msgs = dict()

colorLower = (27, 90, 0)
colorUpper = (38, 255, 255)

def add_msg(msg, name, seq=None):
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg


def get_msgs():
    global msgs
    seq_remove = []  # Arr of sequence numbers to get deleted
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq)  # Will get removed from dict if we find synced msgs pair
        # Check if we have both detections and color frame with this sequence number
        if len(syncMsgs) == 2:  # rgb + depth
            for rm in seq_remove:
                del msgs[rm]
            return syncMsgs  # Returned synced msgs
    return None


###pipeline creation
pipeline = dai.Pipeline()

device = dai.Device()
# Define sources and outputs
rgbCamera = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
# spatialDetectionN = pipeline.create(dai.node.SpatialDetectionNetwork)
xoutrgb = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutrgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("SpatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
rgbCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgbCamera.setIspScale(9, 28)
downscaleColor = True
#if downscaleColor: rgbCamera.setIspScale(2, 3)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        rgbCamera.initialControl.setManualFocus(lensPosition)
except:
    raise
rgbCamera.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
rgbCamera.setPreviewSize(640, 360)
rgbCamera.setPreviewSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = True  # required for depth alignment
subpixel = True

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), 480)

# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.45, 0.45)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
rgbCamera.isp.link(xoutrgb.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

time0, time1, time2 = 0, 0, 0
X, Y, Z, T = [], [], [], []
vx, vy, vz = 0, 0, 0
count = 0
file = open("zdata.txt", 'w')

fontScale = 1

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

with device:
    device.startPipeline(pipeline)
    frame = None
    dframe = None
    depthp = None

    # q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    # depth_q = device.getOutputQueue("depth", maxSize=1, blocking=False)
    depth_p = device.getOutputQueue("SpatialData", maxSize=1, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    # right_q = device.getOutputQueue("mono", maxSize = 1, blocking = False)
    time0 = time.time()

    while True:
        a = None
        for name in ['rgb', 'depth']:
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                add_msg(msg, name)

        synced = get_msgs()
        if synced:
            data = np.ones((300, 300)) * 255
            frame = synced['rgb'].getCvFrame()
            dframe = synced['depth'].getFrame()
            # monof = right_q.tryGet()
            # if monof is not None:
            #     monoFrame = monof.getCvFrame()
            # results = model(monoFrame)
            # img1 = results.render()[0]
            # cv2.imshow("image1",img1)

            # frame = cv2.resize(frame, (640, 480))
            # dframe = depthf.getFrame()
            depthFrameColor = cv2.normalize(dframe, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            depthFrameColor = cv2.rotate(depthFrameColor, cv2.ROTATE_180)


            ########## Revanth gawd's code ####################
            frame = imutils.resize(depthFrameColor, width = 600)
            blurred = cv2.GaussianBlur(frame, (11,11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
            # construct a mask for the color "green", then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask

            mask = cv2.inRange(hsv, colorLower, colorUpper)
            mask = cv2.erode(mask, None, iterations = 2)
            mask = cv2.dilate(mask, (40,40), iterations = 2)
            mask = cv2.medianBlur(mask,ksize = 13)
            # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

            center = None

            # detected_circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,40,param1= 30, param2 = 10,
            #                                 minRadius = 1,maxRadius = 100)
            detected_circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,40,param1= 30, param2 = 10,
                                    minRadius = 1,maxRadius = 100)

            if detected_circles is not None:   
                detected_circles = np.uint16(np.around(detected_circles))

                pt = detected_circles[0,0]

                a,b,radius = pt[0], pt[1], pt[2]

                cv2.circle(frame,(a,b),radius,(255,0,0),2)
                cv2.circle(frame,(a,b),1,(255,0,0),3)

                center = (a,b)


            pts.appendleft(center)


            for i in range(1,len(pts)):
              if pts[i-1] is None or pts[i] is None:
                   continue
            
            
              thickness = int(np.sqrt(10/float(i+1)) * 2.5)
              cv2.line(frame,pts[i],pts[i],(0,0,255),thickness)
              cv2.line(frame, pts[i-1],pts[i],(0,255,0),thickness)
              font = cv2.FONT_HERSHEY_COMPLEX
              cv2.putText(frame,str(center),center,font,0.5,(255,0,0))

        
            #cv2.imshow("Frame",frame)
            #cv2.imshow("Mask",mask)
            # cv2.imshow("Blurred",blurred)   
            print(str(center))
            
            if a is not None:

                p1 = (a-radius/1.5,b-radius/1.5)
                p2 = (a+radius/1.5,b+radius/1.5)
                topLeft.x, topLeft.y = p1
                # print(topLeft.x," ",topLeft.y)
                bottomRight.x, bottomRight.y = p2
                config.roi = dai.Rect(topLeft, bottomRight)
                config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)
                cv2.rectangle(frame, p1, p2, (0, 0, 255))
                cv2.rectangle(depthFrameColor, p1, p2, (255, 255, 255), thickness=4)
                spatialData = depth_p.get().getSpatialLocations()
                for depthData in spatialData:

                    x, y, z = depthData.spatialCoordinates.x / 1000, depthData.spatialCoordinates.y / 1000, depthData.spatialCoordinates.z / 1000
                    file.write(f"{depthData.spatialCoordinates.z}\n")
                    print(
                        f"z = {round(z, 1)}m, y = {round(y, 1)}m x = {round(x, 1)}m")
                    T.append(time.time() - time0)
                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    if count != 0:
                        vx, vy, vz = round((X[-1] - X[-2]) / (T[-1] - T[-2]),1),round((Y[-1] - Y[-2]) / (T[-1] - T[-2]),1), round((
                                    Z[-1] - Z[-2]) / (T[-1] - T[-2]),1)
                    print(f" Vx = {vx}, Vy = {vy}, Vz = {vz}")
                    data = cv2.putText(data, f"x={x}m", (20,20), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA, False)
                    data = cv2.putText(data, f"y={y}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness,
                                    cv2.LINE_AA, False)
                    data = cv2.putText(data, f"z={z}m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness,
                                    cv2.LINE_AA, False)
                    data = cv2.putText(data, f"Vx={vx}m/s", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness,
                                    cv2.LINE_AA, False)
                    data = cv2.putText(data, f"Vy={vy}m/s", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness,
                                    cv2.LINE_AA, False)
                    data = cv2.putText(data, f"Vz={vz}m", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness,
                                    cv2.LINE_AA, False)
                    count += 1
                cv2.imshow("image1", frame)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("data",data)
            # Results
            if cv2.waitKey(1) == ord('q'):
                file.close()
                break
