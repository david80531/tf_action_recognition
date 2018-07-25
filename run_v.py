
# coding: utf-8

# In[1]:


import argparse
import logging
import time
import sys
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0





# In[ ]:


if __name__ == '__main__':
    '''parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='./action_dataset/v_golf_11_01.avi')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()'''
    model_path='mobilenet_thin'
    resolution = '320x240'
    showBG=True
    
    logger.debug('initialization %s : %s' % (model_path, get_graph_path(model_path)))
    w, h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model_path), target_size=(w, h))
    
    
    video = './action_dataset/Basketball/v_Basketball_g01_c01.avi'
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (320, 240))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resize_out_ratio = 8.0
    print("Image Size: %d x %d" % (width, height)) 
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        
        ret_val, image = cap.read()
        #cv2.imshow('tf-pose-estimation input', image)
        if ret_val == True:
          humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
          print (humans)
          if not showBG:
              image = np.zeros(image.shape)
          image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
          cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          cv2.imshow('tf-pose-estimation result', image)
          fps_time = time.time()
          out.write(image)
                
          if cv2.waitKey(1) == 27:
              break
        else:
    cv2.destroyAllWindows()
logger.debug('finished+')