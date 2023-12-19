import torch
import matplotlib.pyplot as plt
import freenect
import cv2
import numpy as np

import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt


def quadratic_model (x,a,b,c):
    return a*x*x+b*x+c
#function to get RGB image from kinect
def get_video():
    #freenect.RES
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    print(array)
    #print(type(array))
    return array

# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5l6',pretrained = True)
#model = torch.load("modelibot.pkl")
pts = []
x_dat =[]
y_dat =[]
while 1:
    img = get_video()  # or file, Path, PIL, OpenCV, numpy, list
    #print('image size=',img.shape)
    results = model(img)
    img1 = results.render()[0]
    cv2.imshow("image",img1)
    depthhh=get_depth()
    cv2.imshow("depth",depthhh)
    #labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    #print(cord_thres)
    df=results.pandas().xyxy[0]
    df=df[df['name']=='sports ball' ]
    if df.empty:
        df=df[df['name']=='frisbee']
        if df.empty:
            df=df[df['name']=='donut']
        

    
    
    xmin=df.iloc[:,0].values
    ymin=df.iloc[:,1].values
    xmax=df.iloc[:,2].values
    ymax=df.iloc[:,3].values
    xmin=(xmin+xmax)/2
    ymin=(ymin+ymax)/2
    namessss=df.iloc[:,6].values
    print(results)
    print(namessss)
    print('X=',xmin)
    print('Y=',ymin)
    x_up=np.ceil(xmin)
    y_up=np.ceil(ymin)
    x_down=np.floor(xmin)
    y_down=np.floor(ymin)
    x_up=list(x_up)
    x_down=list(x_down)
    if(len(x_up) != 0) :
        
        intxxx=int( x_up[0])
        intyyy=int(y_down[0])
        pts.append([intxxx,intyyy])
        x_dat.append(intxxx)
        y_dat.append(intyyy)
        x_data = np.array(x_dat)
        y_data = np.array(y_dat)
        if len(x_data)>2:  
                x_data = x_data[:-2]
                y_data = y_data[:-2]
        arr = np.array(pts)
        arr = arr.reshape((-1,1,2))
        #img = cv2.circle(img, (intxxx,intyyy), radius=1, color=(0, 0, 255), thickness=3)
        img = cv2.polylines(img, [arr],
                      isClosed= False, color = (0,0,255), thickness=3)
        coords = pd.DataFrame({"X-coordinates" : x_data ,"Y-coordinates" : y_data})
        quadratic_curve = np.polyfit(x_data,y_data,2) #Gives best fit values for a b c

#OPTIMAL PARAMETERS
        a_opt , b_opt , c_opt = quadratic_curve
        x_model_quad = np.linspace (min(x_data), max(x_data), 100)
        y_model_quad = quadratic_model(x_model_quad,a_opt,b_opt,c_opt)
        
        curve = np.column_stack((x_model_quad.astype(np.int32), y_model_quad.astype(np.int32)))
        cv2.polylines(img, [curve], False, (0,255,255))
        img2 = img
        cv2.imshow("plot",img2)
    # else:
    #     pts = []
        #z=depthhh[intxxx][intyyy]
        #print(z)
    # print(type(x_down))
    # print(np.shape(x_up))
    # print(x_up-x_down+3)
    # print("lmao "+int(x_up))
    # print(depthhh[x_up][y_up])
    # Z_up=np.zeros(x_up.shape[0])
    # Z_down=np.zeros(x_up.shape[0])
    
    # for i in range(0,len(x_up)):
    #    Z_up[i]=depthhh[x_up[i]][y_up[i]]
    #    Z_down[i]=depthhh[x_down[i]][y_down[i]]

    # Z=(Z_down+Z_up)/2
    # print('Z= ',Z)
    # print(depthhh.shape)
    
    cv2.imshow("ball",img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()