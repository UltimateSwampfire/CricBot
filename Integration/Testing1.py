import cv2
import numpy as np
import depthai as dai

pipeline = dai.Pipeline()
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P



monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
colorCamera = pipeline.create(dai.node.ColorCamera)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
xoutColor = pipeline.create(dai.node.XLinkOut)
xoutColor.setStreamName('color')


#Properties

colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)






# colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
# colorCamera.setIspScale(9,28)


# xoutColor.input.setBlocking(False)
# xoutColor.input.setQueueSize(1)



monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
colorCamera.isp.link(xoutColor.input)

lrcheck = False
subpixel = False


stereo.depth.link(xoutDepth.input)

# stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# stereo.setLeftRightCheck(lrcheck)
# stereo.setSubpixel(subpixel)

with dai.Device(pipeline) as device:
    # depthQueue = device.getOutputQueue(name = "depth", maxSize = 1, blocking = False)
    colorQueue = device.getOutputQueue(name = "color")



    while True:
        # depthFrame = depthQueue.get().getCvFrame()
        colorFrame = colorQueue.get().getCvFrame()

        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)




        cv2.imshow("Depth",depthFrameColor)
        # cv2.imshow("Color", colorFrame)

        print("Color Frame : {0}, Depth Frame : {1}".format(colorFrame.shape,depthFrameColor.shape))

        key = cv2.waitKey(1)

        if key == ord('q'):
            break


cv2.destroyAllWindows()




