import depthai as dai
import cv2


pipeline = dai.Pipeline()

colorCamera = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

#Properties
colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCamera.setVideoSize(426,240)
xoutVideo.input.setQueueSize(1)
xoutVideo.input.getBlocking(False)

#Linking
colorCamera.video.link(xoutVideo.input)

with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name = "video", maxSize = 1, blocking = False)
    while True:
        frame = video.get().getCvFrame()

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xff

        if key == ord("q"):
            break

cv2.destroyAllWindows