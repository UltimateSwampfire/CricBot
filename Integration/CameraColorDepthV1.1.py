from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import depthai as dai

pts = deque(maxlen=10)
msgs = dict()


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
        if len(syncMsgs) == 3:  # rgb + depth
            for rm in seq_remove:
                del msgs[rm]
            return syncMsgs  # Returned synced msgs
    return None


ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help = "path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=16, help="max buffer size")
ap.add_argument("-u", "--hsv", type=str, default='g', help="Ball color")

args = vars(ap.parse_args())

print(args)
# Define lower and upper boundaries of the ball color


if args["hsv"] == "g":  # green
    colorLower = (27, 90, 0)
    colorUpper = (38, 255, 255)
elif args["hsv"] == "y":  # yellow
    colorLower = (22, 132, 72)
    colorUpper = (28, 255, 207)
elif args["hsv"] == "r":  # red
    colorLower = (0, 106, 0)
    colorUpper = (6, 177, 255)
elif args["hsv"] == "o":  # orange
    colorLower = (7, 100, 100)
    colorUpper = (14, 255, 255)

# #Green
# colorLower = (30, 54, 0)
# colorUpper = (52, 209, 255)

pts = deque(maxlen=args["buffer"])

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

colorCamera = pipeline.create(dai.node.ColorCamera)
colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # OAK-D-Lite
#colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)   # OAK-D-S2
# colorCamera.setIspScale(1, 2)  # Comment this for OAK-D-S2
colorCamera.setIspScale(1,3) #For OAK-D-S2
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
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # OAK-D-Lite
# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P) #OAK-D-S2
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # OAK-D-Lite
# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P) #OAK-D-S2
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = True
subpixel = False

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
# stereo.setOutputSize(640,360)
# Config

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
with device:
    device.startPipeline(pipeline)
    frame = None
    dframe = None
    depthp = None

    # Output queue will be used to get the depth frames from the outputs defined above
    # depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    # spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    # colorQueue = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)

    color = (100, 100, 0)
    # time.sleep(2.0)

    while True:
        for name in ['rgb', 'depth', 'spatialData']:
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                add_msg(msg, name)

        synced = get_msgs()
        if synced:
            spatialData = synced['spatialData'].getSpatialLocations()
            frame = synced['rgb'].getCvFrame()
            depthFrame = synced['depth'].getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            wid = np.shape(frame)[0]
            height = np.shape(frame)[1]
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # frame = cv2.resize(frame,(640,400))

            # construct a mask for the color "green", then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask

            mask = cv2.inRange(hsv, colorLower, colorUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # find contours and initialize (x,y) coordso of ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)
            center = None

            if len(cnts) > 0:
                # find largest contour in the mask
                # Use it to compute minimum enclosing circle and centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                # center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
                center = (int(x), int(y))

                if radius > 5:
                    # circle for ball
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    # circle for centroid
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    font = cv2.FONT_HERSHEY_COMPLEX
                    # cv2.putText(frame,str(cv2.contourArea(c)),center,font,0.5,(0,255,0))
                    cv2.putText(frame, str(center), center, font, 0.7, (0, 0, 0))

            pts.appendleft(center)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue

                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)

            # frame_dim = frame.shape
            # print(frame_dim[0])
            # print("(Hellooo,{}".format(frame_dim))

            if center is not None and radius is not None:
                p1 = (int(center[0] - radius / 1.5), int(center[1] - radius / 1.5))
                p2 = (int(center[0] + radius / 1.5), int(center[1] + radius / 1.5))
                topLeft.x, topLeft.y = p1
                # print(topLeft.x," ",topLeft.y)
                bottomRight.x, bottomRight.y = p2

                # topLeft = dai.Point2f(0.2,0.2)
                # bottomRight = dai.Point2f(0.8,0.8)
                # print("FIRST IF")
                # print(frame_dim)

                config.roi = dai.Rect(topLeft, bottomRight)
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)

                # depthFrame values are in millimeters
                # depthFrame = cv2.flip(depthFrame, flipCode=1)
                # combinedFrame = np.array(frame/2 + depthFrameColor/2)
                # cv2.imshow("Combined",combinedFrame)
                for depthData in spatialData:
                    roi = depthData.config.roi
                    # roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                    xmin = int(roi.topLeft().x)
                    ymin = int(roi.topLeft().y)
                    xmax = int(roi.bottomRight().x)
                    ymax = int(roi.bottomRight().y)

                    fontType = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                    cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                                fontType, 0.5, (255, 255, 255))
                    cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                                fontType, 0.5, (255, 255, 255))
                    cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                                fontType, 0.5, (255, 255, 255))
                    print(f"depth is {depthData.spatialCoordinates.z / 1000}m")

            # Show the frame
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("contour", mask)
            cv2.imshow("Color Camera", frame)
            # print("Color : {0} \n Depth : {1}".format(frame.shape, depthFrameColor.shape))
            # print(np.shape(frame))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
