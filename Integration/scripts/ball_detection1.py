#!/usr/bin/env python
import rospy, cv2, cv_bridge
import ros_numpy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import message_filters

xcamera = 0
ycamera = 0
zcamera = 1

ball_coord_pub = rospy.Publisher('ball_coordinate',PointStamped,queue_size=10);
bridge = CvBridge()

def pixelToCoordinate(u,v,pCloud,ball_coord_msg):

    # print(u,v,"pixel coord")
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pCloud,remove_nans=False)
    coord = xyz_array[u][v]

    coord1 = [0,0,0]
    coord1[0] = xcamera + coord[2]
    coord1[1] = ycamera - coord[0]
    coord1[2] = zcamera - coord[1]
    
    t1 = ((pCloud.header.stamp.secs*(10**9)+pCloud.header.stamp.nsecs)/(1000000000.0))
    print("measured coord:",coord1,"time:",t1)#,"actual coord",[3.5-4*t1,0.00,4*t1-(0.5*9.8*(t1*t1))])
         
    #subject to change depending on orientation of camera ....cleanup later
    ball_coord_msg.point.x = xcamera + coord[2]
    ball_coord_msg.point.y = ycamera - coord[0]
    ball_coord_msg.point.z = zcamera - coord[1]

    ball_coord_pub.publish(ball_coord_msg) 
    


def callback(img,pCloud):
    ball_coord_msg = PointStamped()
    ball_coord_msg.header.stamp = img.header.stamp

    time1 = ((img.header.stamp.secs*(10**9)+img.header.stamp.nsecs)/(1000000000.0))
    # print(time1)
    # return

    img = bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
    # cv2.waitKey(3)
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("gray", gray)
    # cv2.waitKey(1)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:        
            pixelToCoordinate(y,x,pCloud,ball_coord_msg)            
                
def listener():
    
    print("ball detection1 node initiated!!")  
    rospy.init_node('ball_detection1')
    image_sub = message_filters.Subscriber('/camera/color/image_raww', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/points', PointCloud2)
    ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()


if __name__ == "__main__":
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

