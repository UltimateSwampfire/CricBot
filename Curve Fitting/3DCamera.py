import depthai as dai
import cv2

pipeline = dai.Pipeline()

mono = pipeline.createMonoCamera()
mono.setBoardSocket(dai.CameraBoardSocket.LEFT)

xout = pipeline.createXLinkOut()
xout.setStreamName("left")
mono.out.link(xout.input)

with dai.Device(pipeline) as device:
    queue = device.getOutputQueue(name = "left")
    frame = queue.get()

    imOut = frame.getCvFrame()
    cv2.imshow("Image",imOut)
