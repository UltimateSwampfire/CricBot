#INCLUDES DETECTION

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import depthai as dai
import matplotlib.pyplot as plt

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
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
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

#Prediction Parameters

# v_assumed = 50 #kmph
# d_assumed = 10 #meters

pts_lst = [] # list of (x,y,z) coords0

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

    closest_depth = 0 #mm
    max_distance = 5000 #mm
    min_distance = 400 #mm

    while True:
        for name in ['rgb', 'depth', 'spatialData']:
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                add_msg(msg, name)

        synced = get_msgs()
        if synced:
            spatialData = synced['spatialData'].getSpatialLocations()
            frame = synced['rgb'].getCvFrame()
            frame = cv2.rotate(frame,cv2.ROTATE_180)
            frame = cv2.flip(frame,flipCode = 1)
            depthFrame = synced['depth'].getFrame()
            depthFrame = cv2.rotate(depthFrame,cv2.ROTATE_180)
            depthFrame = cv2.flip(depthFrame,flipCode=1)
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

            curr_point = (0,0,0)

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
                    
                    curr_point = ((depthData.spatialCoordinates.x),(depthData.spatialCoordinates.y),(depthData.spatialCoordinates.z))
                    # print(f"depth is {round(depthData.spatialCoordinates.z / 1000,2)} m")


                    # #CONDITION WHERE IF BALL IS CLOSER THAN x m ONLY THEN START COLLECTING POINTS

                    # if round(depthData.spatialCoordinates.z) < max_distance:
                    #     print("Depth : %0.2f" %depthData.spatialCoordinates.z)

                    #     # print(f"depth is {round(depthData.spatialCoordinates.z / 1000,2)} m")

                    #     pts_lst.append((depthData.spatialCoordinates.x/1000,
                    #                         depthData.spatialCoordinates.y/1000,
                    #                         depthData.spatialCoordinates.z/1000))  #In millimeters
                        
                    #     # if int(depthData.spatialCoordinates.z) < min_distance:
                    #     #     closest_depth = depthData.spatialCoordinates.z
                    #     #     break

                    # START COLLECTING POINTS AFTER PRESSING A KEY:

                    # if cv2.waitKey(1) & 0xff == ord('s'):

                    #     print("Hello!! You just pressed S!")

            
            # Show the frame
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("Color Camera", frame)
            # cv2.imshow("contour", mask)
            # print("Color : {0} \n Depth : {1}".format(frame.shape, depthFrameColor.shape))
            # print(np.shape(frame))
            key = cv2.waitKey(1)

            print("Depth : %0.2f"%curr_point[2])

            if curr_point != (0,0,0):

                if  curr_point[2] < max_distance:

                    print("POINT WITHIN RANGE.")
                    pts_lst.append(curr_point)
                    if curr_point[2] < min_distance:
                        print("MIN DISTANCE REACHED!") #In millimeters
                        # break
            


            if key == ord('q'):
                break


    cv2.destroyAllWindows()


print(len(pts_lst))

for item in pts_lst:
    print(item)
#Prediction Algorithm

# def linear_model (x,m): #Where c = 0 in y=mx+c
#     return m*x

def quadratic_model(x, a, b, c):
    return a*x*x+b*x+c

#Linear regression

#Defining parameters

#Data points wrt camera

x_data = np.array([tup[0] for tup in pts_lst])
y_data = np.array([tup[1] for tup in pts_lst])
z_data = np.array([tup[2] for tup in pts_lst])

# # print(x_data)

# #First detected point
# x0 = x_data[0]
# y0 = y_data[0]
# z0 = z_data[0]

# #camera coords
# x_cam = -x0
# y_cam = -y0
# z_cam = -z0

# #set every point in reference to first point

# x_rel = np.array([x - x0 for x in x_data])
# y_rel = np.array([y - y0 for y in y_data])
# z_rel = np.array([z - z0 for z in z_data])




# # xs , ys , xys = 0 , 0 , 0

# # # for i in range(0,len(pts_lst)):
# # #         xs+=np.square(x_rel[i])
# # #         ys+=np.square(y_rel[i])
# # #         xys+=x_rel[i]*y_rel[i]

# # xs = np.sum(x_rel**2)
# # ys = np.sum(y_rel**2)
# # xys = np.sum(x_rel * y_rel)


# # m = (-(xs-ys)+np.sqrt((xs-ys)**2+4*(xys)**2))/(2*xys) 
# # m = (-(xs-ys)+np.sqrt((xs-ys)**2-4*(xys)**2))/(2*xys) 

# m, c = np.polyfit(x_rel,y_rel,1)

# #Shifting datapoints to linear curve

# # for i in range(0,len(pts_lst)):
# #         x_rel[i] = (x_rel[i]+m*y_rel[i])/(m**2+1)
# #         y_rel[i] = m*x_rel[i]

# # x_rel = (x_rel + m * y_rel) / (m**2 + 1)
# # y_rel = m * x_rel

# # r_rel = [np.sqrt(x_rel[i]**2 + y_rel[i]**2) for i in range(len(pts_lst))]

# r_rel = np.sqrt(x_rel**2 + y_rel**2)


# #QUADRATIC REGRESSION

# a , b , c = np.polyfit(r_rel,z_rel,2)


# #Final predicted point

# #How far ahead we want bat to go:

# # y_up_front = 0.5 #meters, i.e. get the coordinates when ball will be 0.5m in front of cam

# y_up_front = 500 #milimeters, i.e. get the coordinates when ball will be 0.5m in front of cam



# #up_front plane

# y_up = y_cam + y_up_front

# #Coords in first_point frame

# xfp = y_up/m
# yfp = y_up
# rfp = np.sqrt(xfp**2 + yfp**2)
# zfp = quadratic_model(rfp, a, b, c)


# #Coords in camera frame

# #For camera, y is height and z is depth
# #But this calc takes height as z and y as depth
# #Before submitting to IK, just interchange y and z coords

# xf = round(xfp - x_cam,2) #X 
# zf = round(yfp - y_cam,2) #Depth
# yf = round(zfp - z_cam,2) #Height

# #Final coords
# final_coords = (xf,yf,zf) #milimeters


# print("Final_coords are : ({} m, {} m, {} m) ".format(round(xf/1000,2),round(yf/1000,2),zf/1000))


# #plotting values:

# x_plot = np.arange(-2000,2000,10)
# y_plot =  m * x_plot
# r_plot = np.sqrt(x_plot**2 + y_plot**2)
# z_plot = quadratic_model(r_plot,a,b,c)

# fig, ax = plt.subplots(2,1, figsize = (5,10))

# ax[0].scatter(x_data,z_data)
# ax[0].plot(x_plot-x_cam,y_plot-y_cam)

# plt.grid()

# ax[0].set_xlabel("X-axis")
# ax[0].set_ylabel("Z-axis")
# ax[0].set_title("Top View (Straight line)")

# ax[1].scatter(z_data,y_data)
# # ax[1].plot(y_plot + y_cam,z_plot+z_cam)

# ax[1].set_xlabel("X-rel")
# ax[1].set_ylabel("Z-rel")
# ax[1].set_title("Parabolic Curve (r vs z)")

# plt.grid()

# # ax1 = fig.add_subplot(111, projection = '3d')
# # ax1.scatter(x_data,y_data,z_data)

# plt.show()

#First detected point
x0 = x_data[0]
y0 = y_data[0]
z0 = z_data[0]

#camera coords
x_cam = -x0
y_cam = -y0
z_cam = -z0

#set every point in reference to first point

# x_rel = [x - x0 for x in x_data]
# y_rel = [y - y0 for y in y_data]
# z_rel = [z - z0 for z in z_data]

x_rel = x_data - x0
y_rel = y_data - y0
z_rel = z_data - z0

xs , ys , xys = 0 , 0 , 0

# for i in range(0,len(pts_lst)):
#         xs+=np.square(x_rel[i])
#         ys+=np.square(y_rel[i])
#         xys+=x_rel[i]*y_rel[i]

xs = np.sum(np.square(x_rel))
ys = np.sum(np.square(y_rel))
xys = np.sum(x_rel * y_rel)


m = (-(xs-ys)+np.sqrt((xs-ys)**2+4*(xys)**2))/(2*xys) 


#Shifting datapoints to linear curve

# for i in range(0,len(pts_lst)):
#         x_rel[i] = (x_rel[i]+m*y_rel[i])/(m**2+1)
#         y_rel[i] = m*x_rel[i]


x_rel = (x_rel + m * y_rel)/(m**2 + 1)
y_rel = m * x_rel

r_rel = np.sqrt(x_rel ** 2 + y_rel ** 2)

#QUADRATIC REGRESSION

a , b , c = np.polyfit(r_rel,z_rel,2)


#Final predicted point

#How far ahead we want bat to go:

y_up_front = 0.5 #meters, i.e. get the coordinates when ball will be 0.5m in front of cam


#up_front plane

y_up = y_cam + y_up_front

#Coords in first_point frame

xfp = y_up/m
yfp = y_up
rfp = np.sqrt(xfp**2 + yfp**2)
zfp = quadratic_model(rfp, a, b, c)


#Coords in camera frame

#For camera, y is height and z is depth
#But this calc takes height as z and y as depth
#Before submitting to IK, just interchange y and z coords

xf = round(xfp - x_cam,2) #X 
zf = round(yfp - y_cam,2) #Depth
yf = round(zfp - z_cam,2) #Height

#Final coords
final_coords = (xf,yf,zf)


print("Final_coords are : ({} m, {} m, {} m) ".format(xf,yf,zf))

x_plot = np.arange(min(x_data)*2,max(x_data),0.1)
y_plot= x_plot * m
r_plot = np.sqrt(x_plot**2 + y_plot **2)
z_plot = quadratic_model(r_plot,a,b,c)

x_plot -= x_cam
y_plot -= y_cam
z_plot -= z_cam


fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x_data,z_data,y_data)
# ax.plot(x_plot,y_plot,z_plot)
ax.set_xlabel("X-axis")
ax.set_ylabel("Z-axis")
ax.set_zlabel("Y-axis")
plt.show()



