# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:16:07 2015

@author: dlef

***** Warning!!! Disable USB3 ports in BIOS before using the Xtion*****
"""
#
#from primesense import openni2
#
#openni2.initialize('/home/dlef/apps_and_libs/asus_xtion/OpenNI2-master/Packaging/Final/OpenNI-Linux-x86-2.2/Redist/')     # can also accept the path of the OpenNI redistribution
#
#dev = openni2.Device.open_any()
##print dev.get_sensor_info()
#
#depth_stream = dev.create_depth_stream()
#depth_stream.start()
#frame = depth_stream.read_frame()
#frame_data = frame.get_buffer_as_uint16()
#depth_stream.stop()0
#
#openni2.unload()



import cv2
import cv2.cv as cv
import numpy as np
from time import strftime

rec= True
#rec= False
savepath= '/home/dlef/Desktop/shuttledetect/testing data/'
fname= ''
rate = 1     # in msec

def convert2uint8(image):
    '''convert and scale an image to unsigned 8 bits for display'''
    
    im_range = np.max(image)-np.min(image)
    im_offseted = image - np.min(image)
    imuint8= (im_offseted / float(im_range))*255    # be careful to make calc in floats
    
    return imuint8.astype('uint8')

capture = cv2.VideoCapture(cv.CV_CAP_OPENNI)
capture.set(cv.CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv.CV_CAP_OPENNI_VGA_30HZ)

print capture.get(cv.CV_CAP_PROP_OPENNI_REGISTRATION)
######################################################################################
# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output_Xtion.avi',fourcc, 20.0, (640,480))


## Release everything if job is finished
#capture.release()
#out.release()
######################################################################################
depth_map_list= []
valid_depth_mask_list= []
disp_map_list= []
image_list= []
while True:
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
#    
    depth_map= convert2uint8(depth_map)
    cv2.imshow("depth camera", depth_map)
    cv2.imshow("valid depth mask", valid_depth_mask)
    cv2.imshow("disparity map", disp_map)
    cv2.imshow("point cloud", point_cloud)
    cv2.imshow("rgb camera", image)
    ####################################
    print rec
    if rec:
#        image = cv2.flip(image,0)

        # write the flipped frame
        out.write(image)

        cv2.imshow('image',image)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    else:
#        break
    ####################################
    if rec:
        depth_map_list.append(depth_map)
        valid_depth_mask_list.append(valid_depth_mask)
        disp_map_list.append(disp_map)
        image_list.append(image)
    
    keypress= cv2.waitKey(rate)%256    
    if keypress == 27:
        break

if rec:
    datetime= strftime('%Y%m%d_%H%M%S-')
    for i in range(len(image_list)):
#        cv2.imwrite(savepath+datetime+fname+str(i)+'-depthmap.tiff', depth_map_list[i])
#        cv2.imwrite(savepath+datetime+fname+str(i)+'-validmask.tiff', valid_depth_mask_list[i])
#        cv2.imwrite(savepath+datetime+fname+str(i)+'-dispmap.tiff', disp_map_list[i])
        cv2.imwrite(savepath+datetime+fname+str(i)+'-rgb.tiff', image_list[i])
    
cv2.destroyAllWindows()
capture.release()
out.release()