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
from scipy.optimize import curve_fit
from calc import HostSpatialsCalc
import math

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
        if len(syncMsgs) == 2:  # rgb + depth
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
#spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

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
#xoutSpatialData = pipeline.create(dai.node.XLinkOut)
#xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
colorCamera.isp.link(xoutRGB.input)
# colorCamera.preview.link(xoutRGB.input)


xoutDepth.setStreamName("depth")
#xoutSpatialData.setStreamName("spatialData")
#xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        colorCamera.initialControl.setManualFocus(lensPosition)
except:
    raise
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

#config = dai.SpatialLocationCalculatorConfigData()
#config.depthThresholds.lowerThreshold = 100
#config.depthThresholds.upperThreshold = 10000
#config.roi = dai.Rect(topLeft, bottomRight)

#spatialLocationCalculator.inputConfig.setWaitForMessage(False)
#spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
#spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
#stereo.depth.link(spatialLocationCalculator.inputDepth)

#spatialLocationCalculator.out.link(xoutSpatialData.input)
#xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
with device:
    device.startPipeline(pipeline)
    frame = None
    dframe = None
    depthp = None

    # Output queue will be used to get the depth frames from the outputs defined above
    # depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    # spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)
    #spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    # colorQueue = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)

    color = (100, 100, 0)
    # time.sleep(2.0)

    closest_depth = 0 #mm
    max_distance = 4000 #mm
    min_distance = 1000 #mm
    hostSpatials = HostSpatialsCalc(device)
    delta = 2
    hostSpatials.setDeltaRoi(delta)
    while True:
        for name in ['rgb', 'depth']:
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                add_msg(msg, name)

        synced = get_msgs()
        if synced:
           # spatialData = synced['spatialData'].getSpatialLocations()
            frame = synced['rgb'].getCvFrame()
            # frame = cv2.rotate(frame,cv2.ROTATE_180)
            # frame = cv2.flip(frame,flipCode = 0)
            depthData = synced['depth']
            depthFrame = depthData.getFrame()

            # depthFrame = cv2.rotate(depthFrame,cv2.ROTATE_180)
            # depthFrame = cv2.flip(depthFrame,flipCode=0)
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
                spatials, centroid = hostSpatials.calc_spatials(depthData, center)

                # topLeft = dai.Point2f(0.2,0.2)
                # bottomRight = dai.Point2f(0.8,0.8)
                # print("FIRST IF")
                # print(frame_dim)

                # config.roi = dai.Rect(topLeft, bottomRight)
                # cfg = dai.SpatialLocationCalculatorConfig()
                # cfg.addROI(config)
                # spatialCalcConfigInQueue.send(cfg)

                # depthFrame values are in millimeters
                # depthFrame = cv2.flip(depthFrame, flipCode=1)
                # combinedFrame = np.array(frame/2 + depthFrameColor/2)
                # cv2.imshow("Combined",combinedFrame)

               

                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cv2.rectangle(depthFrameColor,p1, p2, color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                if  math.isnan(spatials["x"]) is False:
                    cv2.putText(depthFrameColor, f"X: {int(spatials['x'])} mm", (p1[0]+ 10, p1[1] + 20),
                                fontType, 0.5, (255, 255, 255))
                    cv2.putText(depthFrameColor, f"Y: {int(spatials['y'])} mm", ( p1[0]+ 10, p1[1] + 35),
                                fontType, 0.5, (255, 255, 255))
                    cv2.putText(depthFrameColor, f"Z: {int(spatials['z'])} mm", (p1[0] + 10, p1[1] + 50),
                                fontType, 0.5, (255, 255, 255))
                    
                    curr_point = ((spatials['x']),(spatials['y']),(spatials['z']))
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
            # cv2.imshow("depth", depthFrameColor)
            # cv2.imshow("Color Camera", frame)
            cv2.imshow("Frame",np.hstack((depthFrameColor,frame)))
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
                        break
            


            if key == ord('q'):
                break


    cv2.destroyAllWindows()


###############################################

print(len(pts_lst))

for item in pts_lst:
    print(item)

#PREDICTION ALGORITHM

# def parabolic(x, a, b ,c):

#     return a*x*x + b*x + c


def plane(x,x0,z0,u0,v0):
    
    t = (x-x0)/u0
    z = z0 - v0*t
    return z

def inv_plane(z,z0,x0,v0,u0):
    t = (z0 - z) / v0
    x = u0 * t + x0
    return x


def profile(z,z0,y0,v0,w0):

    t = (z0 - z)/v0
    # y = y0 + w0*t - 0.5 * 9806.65 * t * t
    y = y0 + w0*t - 0.5 * 9.81 * t * t

    return y
    

#in milimeters
# x_data = np.array([tup[0] for tup in pts_lst])
# y_data = np.array([tup[1] for tup in pts_lst]) #Global Height (vertical)
# z_data = np.array([tup[2] for tup in pts_lst]) #Global Depth (along pitch)

#in meters
x_data = np.array([round(tup[0]/1000,2) for tup in pts_lst])
y_data = np.array([round(tup[1]/1000,2) for tup in pts_lst]) #Global Height (vertical)
z_data = np.array([round(tup[2]/1000,2) for tup in pts_lst]) #Global Depth (along pitch)


#Calculating parameters

constants_1 = curve_fit(plane,x_data, z_data)
x1_fit = constants_1[0][0]
z1_fit = constants_1[0][1]
vx1_fit = constants_1[0][2]
vz1_fit = constants_1[0][3]
print("Constants 1 : ",constants_1[0])

constants_2 = curve_fit(profile,z_data,y_data)
z2_fit = constants_2[0][0]
y2_fit = constants_2[0][1]
vz2_fit = constants_2[0][2]
vy2_fit = constants_2[0][3]
print("Constants 1 : ",constants_2[0])


x_parametric = np.arange(min(x_data),max(x_data),1000)

t = (x_parametric - x1_fit) / vx1_fit

z_parametric = z1_fit + vz1_fit * t

t = (z_parametric - z2_fit) / vz2_fit

# y_parametric = y2_fit + vy2_fit * t - 0.5 * 9806.65 * t * t
y_parametric = y2_fit + vy2_fit * t - 0.5 * 9.81 * t * t



z_final = 0.1 #meters

x_final = inv_plane(z_final,z1_fit,x1_fit,vz1_fit,vx1_fit)

y_final = profile(z_final,z2_fit,y2_fit,vz2_fit,vy2_fit)



# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# # ax.view_init(0,2)
# # plt.xlim([-2,2])
# # plt.ylim([0,20])

# ax.scatter3D(x_data, z_data, y_data, 'greens')
# ax.plot(x_parametric,z_parametric,y_parametric)
# ax.set_xlabel("Width")
# ax.set_ylabel("Depth")
# ax.set_zlabel("Height")

# plt.show()


print("Final coordinates are : ")
print("{} m,{} m,{} m".format(round(x_final,2), round(y_final,2), z_final))
fig, ax = plt.subplots(2,2)



ax[0][0].scatter(x_data,z_data)
ax[1][0].plot(x_parametric,z_parametric,linestyle='dotted')
ax[0][0].set_xlabel("X-axis / Width")
ax[0][0].set_ylabel("Z-axis / Depth")
ax[0][0].set_xlim(min(x_data),max(x_data))

ax[0][1].scatter(z_data,y_data)
ax[1][1].plot(z_parametric,y_parametric,linestyle='dotted')
ax[0][1].set_xlabel("Z-axis / Depth")
ax[0][1].set_ylabel("Y-axis / Height")
ax[0][1].set_xlim(min(z_data),max(z_data))




plt.show()