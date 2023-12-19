# INCLUDES DETECTION

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


start_time = time.time()

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

pts = deque(maxlen=args["buffer"])

# Prediction Parameters

# v_assumed = 50 #kmph
# d_assumed = 10 #meters

pts_lst = []  # list of (x,y,z) coords0
# in meters
x_data= []
y_data= []
z_data= []


# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

colorCamera = pipeline.create(dai.node.ColorCamera)
colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # OAK-D-Lite
# colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)   # OAK-D-S2
# colorCamera.setIspScale(1, 2)  # Comment this for OAK-D-S2
colorCamera.setIspScale(1, 3)  # For OAK-D-S2
# colorCamera.setVideoSize(1920,1080)
# colorCamera.initialControl.setSharpness(0)
# colorCamera.setPreviewSize(1920,1080)

xoutRGB = pipeline.create(dai.node.XLinkOut)
xoutRGB.setStreamName("rgb")
xoutRGB.input.setBlocking(False)
xoutRGB.input.setQueueSize(1)

xoutDepth = pipeline.create(dai.node.XLinkOut)
colorCamera.isp.link(xoutRGB.input)

xoutDepth.setStreamName("depth")
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

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

with device:
    device.startPipeline(pipeline)
    frame = None
    dframe = None
    depthp = None
    downward = 0
    upward = 0
    offset = 1
    # falseDetec = False
    color = (100, 100, 0)
    closest_depth = 0  # mm
    max_distance = 4000  # mm
    min_distance = 1000  # mm

    throw_distance = 10 #meters
    curr_Z_velocity = 6 #m/s
    impact_time =  throw_distance / curr_Z_velocity #seconds

    g = 9.81 #m/s
    e = 0.82

    actuator_time = 0.4 #seconds
    inv_k_time = 0.2 #seconds
    arduino_time = 0.4 #seconds
    time_tax = actuator_time + inv_k_time + arduino_time

    z_final = 0.1  # meters
    FPS = 30 #FPS

    time_remaining = impact_time - time_tax


    def falseDetec(y_previous,y_current):

        if y_previous < y_current: #if prev point is closer to cam than current point, false
            return True
        else:
            return False

    def get_Z_velocity(z0,z1):
        Zvelocity = ((z1 - z0)/1000)/(1/FPS) #meters / second
        return Zvelocity
    
    def get_Y_velocity(y0,y1):
        Yvelocity = ((y1 - y0)/1000)/(1/FPS) #meters / second
        return Yvelocity
    
    def getImpactTime(z0,z1):
        
        z_velocity = get_Z_velocity(z0,z1)
        min_velocity = 1
        if z_velocity > min_velocity:
            t = (z1/1000 - z_final) / z_velocity #seconds
            return t
        else:
            #ball is close to stationary
            
            return impact_time
    
    def get_bounce_number(current_velocity,current_height,impact_time,initial_time):

        #Assuming at least one bounce is predicted to happen

        u = current_velocity
        t_impact = impact_time
        t_initial = initial_time
        del_t = t_impact - t_initial

        d = current_height
        v0 = np.sqrt(u**2 + 2 * g * d)

        def S(n):
            Sn =  (2 * v0 * e / g) * ((1 + e**n)/(1-e))
            return Sn
        
        n = 2
    
        while True:

            if S(n) <= del_t and S(n+1) >= del_t :
                #n number of bounces happen
                return (n,S(n))

            else:
                n = n+1

    
    def get_final_Y(current_velocity,current_height,impact_time):

        u = current_velocity #m/s
        t_impact = impact_time #seconds
        d = current_height / 1000
        v0 = np.sqrt(u**2 + 2 * g * d)
        t_initial = (u + v0)/g
        t_first_bounce = (2 * v0 * e) / g
        y_final = 0 #SET TO DEFAULT VALUE 

        if t_impact <= t_initial : #No bounce case:

            y_final = d + u * t_impact - 0.5 * g * (t_impact **2)
        
        elif t_impact <= t_initial + t_first_bounce :#Only one bounce

            t_final = t_impact - t_initial

            y_final = e * v0 * t_final - 0.5 * g * (t_final**2)

        elif t_impact > t_initial + t_first_bounce: #2 or more bounces

            (n, Sn) = get_bounce_number(u,d,t_impact,t_initial)
            
            t_final = t_impact - t_initial - Sn

            y_final = (v0 * (e**n)) * t_final - 0.5 * g * (t_final **2) #After 1 bounce, v is v0 * e^1, after n bounces, v is v0 * e^n

        return y_final
    
    def ballLineModel(z,m,x0):

        x = m * z + x0
        return x
    
    # def get_final_X(x_data,z_data):

    #     constants = curve_fit(ballLineModel,z_data,x_data)

    #     m = constants[0][0]
    #     x0 = constants[0][1]

    #     return (x0/1000) #meters

    def get_final_X(x_data,z_data):
        
        constants = np.polyfit(z_data,x_data,1)

        m = constants[0]
        x0 = constants[1]

        x_final = (m * (z_final * 1000) + x0)/1000

        return x_final #meters
    
    







    
    hostSpatials = HostSpatialsCalc(device)
    delta = 2
    hostSpatials.setDeltaRoi(delta)
    # while True:
    # while time_remaining >= 0:
    while True:
        for name in ['rgb', 'depth']:
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                add_msg(msg, name)

        synced = get_msgs()
        if synced:

            frame = synced['rgb'].getCvFrame()
            depthData = synced['depth']
            depthFrame = depthData.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            wid = np.shape(frame)[0]
            height = np.shape(frame)[1]
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
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
                center = (int(x), int(y))

                if radius > 5:
                    # circle for ball
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    # circle for centroid
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(frame, str(center), center, font, 0.7, (0, 0, 0))

            pts.appendleft(center)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue

                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)

            curr_point = (0, 0, 0)

            if center is not None and radius is not None:
                p1 = (int(center[0] - radius / 1.5), int(center[1] - radius / 1.5))
                p2 = (int(center[0] + radius / 1.5), int(center[1] + radius / 1.5))
                spatials, centroid = hostSpatials.calc_spatials(depthData, center)

                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cv2.rectangle(depthFrameColor, p1, p2, color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                if math.isnan(spatials["x"]) is False:
                    cv2.putText(depthFrameColor, f"X: {int(spatials['x'])} mm", (p1[0] + 10, p1[1] + 20),
                                fontType, 0.5, (255, 255, 255))
                    cv2.putText(depthFrameColor, f"Y: {int(spatials['y'])} mm", (p1[0] + 10, p1[1] + 35),
                                fontType, 0.5, (255, 255, 255))
                    cv2.putText(depthFrameColor, f"Z: {int(spatials['z'])} mm", (p1[0] + 10, p1[1] + 50),
                                fontType, 0.5, (255, 255, 255))

                    curr_point = ((spatials['x']), (spatials['y']), (spatials['z']))

            cv2.imshow("Frame", np.hstack((depthFrameColor, frame)))
            key = cv2.waitKey(1)

            # print("Depth : %0.2f" % curr_point[2])
            if time.time() - start_time > 10:
                if curr_point != (0, 0, 0):

                    
                    #APPENDING THE POINTS HAPPENS HERE

                    if curr_point[2] < max_distance and curr_point[2] > min_distance : #ball within range no false detect
                        pts_lst.append(curr_point)   #millimeters
                        
                        x_data.append(curr_point[0]) #millimeters
                        y_data.append(curr_point[1]) #millimeters
                        z_data.append(curr_point[2]) #millimeters
                            

                    #FINAL POINT CACULATIONS

                    if len(pts_lst) >= 2:

                        
                            
                        prev_point = pts_lst[-2]
                        # if not falseDetec(prev_point[2],curr_point[2]) :
                        if True:

                            curr_vz = get_Z_velocity(prev_point[2],curr_point[2])
                            impact_time = getImpactTime(prev_point[2],curr_point[2])
                            
                            # print("Prev point : {} Current Point : {}".format(prev_point[2],curr_point[2]))
                            print("Z velocity : {} m/s, Impact Time : {} seconds".format(round(curr_vz,2),round(impact_time,2)))
                            
                            # curr_vy = get_Y_velocity(prev_point[1],curr_point[1])

                            # x_final = get_final_X(x_data,z_data)
                            # y_final = get_final_Y(curr_vy,curr_point[1],impact_time)

                            # print("Final coordinates are : ({} m,{} m,{} m)".format(round(x_final, 2), round(y_final, 2), z_final))

                            time_remaining = impact_time - time_tax #update time_remaining before loop closes
                            if time_remaining < 0 :
                                print("TIMES OUT! time remanining is : %0.2f seconds"%time_remaining)


                            
                if key == ord('q'):
                    break

    cv2.destroyAllWindows()

###############################################

print("Detection Finished. Number of points collected : ",len(pts_lst))

for item in pts_lst:
    print(item)

# in meters
x_data = np.array([round(tup[0] / 1000, 2) for tup in pts_lst])  #Global Width
y_data = np.array([round(tup[1] / 1000, 2) for tup in pts_lst])  # Global Height (vertical)
z_data = np.array([round(tup[2] / 1000, 2) for tup in pts_lst])  # Global Depth (along pitch)


# time_plot = np.arange(0,len(pts_lst),1)

fig, ax = plt.subplots(2, 2)

ax[0][0].scatter(x_data, z_data)
ax[0][0].set_xlabel("X-axis / Width")
ax[0][0].set_ylabel("Z-axis / Depth")
ax[0][0].set_xlim(min(x_data), max(x_data))

ax[0][1].scatter(z_data, y_data)
ax[0][1].set_xlabel("Z-axis / Depth")
ax[0][1].set_ylabel("Y-axis / Height")
ax[0][1].set_xlim(min(z_data), max(z_data))

# # fig = plt.figure()
# # ax = plt.axes(projection = '3d')
# # # ax.view_init(0,2)
# # # plt.xlim([-2,2])
# # # plt.ylim([0,20])

# # ax.scatter3D(x_data, z_data, y_data, 'greens')
# # ax.plot3D(x_parametric,z_parametric,y_parametric)
# # ax.set_xlabel("Width")
# # ax.set_ylabel("Depth")
# # ax.set_zlabel("Height")

plt.tight_layout()



plt.show()
