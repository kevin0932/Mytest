# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:30:26 2015

@author: qixuanzhang
"""


import cv2
#import cv2.cv as cv
import numpy as np
from time import strftime
import os
import cv
import glob
import fnmatch
import matplotlib.pyplot as plt
import matplotlib as ml
import pylab

rate = 1     # in msec
framecnt = 120
def convert2uint8(image):
    '''convert and scale an image to unsigned 8 bits for display'''
    
    im_range = np.max(image)-np.min(image)
    im_offseted = image - np.min(image)
    imuint8= (im_offseted / float(im_range))*255    # be careful to make calc in floats
    
    return imuint8.astype('uint8')

capture = cv2.VideoCapture(cv.CV_CAP_OPENNI)
capture.set(cv.CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv.CV_CAP_OPENNI_VGA_30HZ)

print capture.get(cv.CV_CAP_PROP_OPENNI_REGISTRATION)

##################################################################################

datetime= strftime('%Y%m%d_%H%M%S')
print datetime
if not os.path.exists('ROI_Depth'):
    os.mkdir('ROI_Depth')
if not os.path.exists('ROI_RGB'):
    os.mkdir('ROI_RGB')
if not os.path.exists('XtionCam'):
    os.mkdir('XtionCam')
if not os.path.exists('RawDepthData'):
    os.mkdir('RawDepthData')
#os.chdir('XtionCam')
#os.mkdir(datetime)
#os.chdir(datetime)
if not os.path.exists('XtionCam/'+str(datetime)):
    os.mkdir('XtionCam/'+str(datetime))
if not os.path.exists('RawDepthData/'+str(datetime)):
    os.mkdir('RawDepthData/'+str(datetime))
if not os.path.exists('XtionCam/'+str(datetime)+'/PointCloud_images'):
    os.mkdir('XtionCam/'+str(datetime)+'/PointCloud_images')
if not os.path.exists('XtionCam/'+str(datetime)+'/ValidDepthMask_images'):
    os.mkdir('XtionCam/'+str(datetime)+'/ValidDepthMask_images')
if not os.path.exists('XtionCam/'+str(datetime)+'/Disparity_images'):
    os.mkdir('XtionCam/'+str(datetime)+'/Disparity_images')
if not os.path.exists('XtionCam/'+str(datetime)+'/RGB_images'):
    os.mkdir('XtionCam/'+str(datetime)+'/RGB_images')
if not os.path.exists('XtionCam/'+str(datetime)+'/Depth_images'):
    os.mkdir('XtionCam/'+str(datetime)+'/Depth_images')

##################################################################################
# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output_Xtion.avi',fourcc, 20.0, (640,480))

frame_depth_3d_stack = np.zeros((480, 640, framecnt))
frame_valid_depth_mask_3d_stack = np.zeros((480, 640, framecnt))

frame_dispartiy_3d_stack = np.zeros((480, 640, framecnt))
frame_point_cloud_4d_stack = np.zeros((480, 640, 3, framecnt))
##################################################################################################################

for i in range(0, framecnt):
    if not capture.grab():
        print "Unable to Grab Frames from camera"
        break
    okay1, depth_map = capture.retrieve(0, cv.CV_CAP_OPENNI_DEPTH_MAP)
    if not okay1:
        print "Unable to Retrieve Depth Map from camera"
        break
    okay3, valid_depth_mask = capture.retrieve(0, cv.CV_CAP_OPENNI_VALID_DEPTH_MASK)
    if not okay3:
        print "Unable to Retrieve Valid depth Mask from camera"
        break
    okay4, disp_map = capture.retrieve(0, cv.CV_CAP_OPENNI_DISPARITY_MAP)
    if not okay4:
        print "Unable to Retrieve Disparity Map from camera"
        break
    okay5, point_cloud = capture.retrieve(0, cv.CV_CAP_OPENNI_POINT_CLOUD_MAP)
    if not okay5:
        print "Unable to Retrieve Point Cloud from camera"
        break
    okay2, image = capture.retrieve(0, cv.CV_CAP_OPENNI_BGR_IMAGE)
    if not okay2:
        print "Unable to retrieve BGR Image from device"
        break
  
    if i==0:
        frame_depth_3d_stack= depth_map
    else:
        frame_depth_3d_stack= np.dstack((frame_depth_3d_stack, depth_map))
        
    if i==0:
        frame_valid_depth_mask_3d_stack= depth_map
    else:
        frame_valid_depth_mask_3d_stack= np.dstack((frame_valid_depth_mask_3d_stack, valid_depth_mask))
    
    
    if i==0:
        frame_dispartiy_3d_stack= depth_map
    else:
        frame_dispartiy_3d_stack= np.dstack((frame_dispartiy_3d_stack, disp_map))

    frame_point_cloud_4d_stack[:,:,:,i]= point_cloud
    
    out.write(image)
    cv2.imshow('Recorded Image Video',image)
        
    depth_map= convert2uint8(depth_map)
    point_cloud= convert2uint8(point_cloud)
#    cv2.imshow("depth camera", depth_map)
#    cv2.imshow("valid depth mask", valid_depth_mask)
#    cv2.imshow("disparity map", disp_map)
#    cv2.imshow("point cloud", point_cloud)
#    cv2.imshow("rgb camera", image)   

    cv2.imwrite('XtionCam/'+str(datetime)+'/PointCloud_images/'+str(i)+'-PointCloud.tiff', point_cloud) 
    cv2.imwrite('XtionCam/'+str(datetime)+'/Disparity_images/'+str(i)+'-Disparity.tiff', disp_map)   
    cv2.imwrite('XtionCam/'+str(datetime)+'/ValidDepthMask_images/'+str(i)+'-ValidDepthMask.tiff', valid_depth_mask) 
    cv2.imwrite('XtionCam/'+str(datetime)+'/Depth_images/'+str(i)+'-Depth.tiff', depth_map)   
    cv2.imwrite('XtionCam/'+str(datetime)+'/RGB_images/'+str(i)+'-RGB.tiff', image)        
        
#     
    keypress= cv2.waitKey(rate)%256    
    if keypress == 27:
        break
#   
## Release everything if job is finished
cv2.destroyAllWindows()
capture.release()
#plt.clf()
out.release()
##################################################################################################################
np.save('RawDepthData/'+str(datetime)+'/frame_depth_3d_stack.npy', frame_depth_3d_stack) 
np.save('RawDepthData/'+str(datetime)+'/frame_valid_depth_mask_3d_stack.npy', frame_valid_depth_mask_3d_stack) 

np.save('RawDepthData/'+str(datetime)+'/frame_dispartiy_3d_stack.npy', frame_dispartiy_3d_stack) 
np.save('RawDepthData/'+str(datetime)+'/frame_point_cloud_4d_stack.npy', frame_point_cloud_4d_stack) 
##################################################################################################################