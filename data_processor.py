
# coding: utf-8

# In[1]:


import argparse
import logging
import time
import sys
import cv2
import numpy as np
import pickle
import random
import os
import psutil
from memory_profiler import profile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from tf_pose.estimator import BodyPart
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


# In[2]:


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
fps_time = 0
num_class = 9
video_dict = { 'PlayingCello':0,'PlayingDaf':1,
                  'PlayingDhol':2,'PlayingFlute':3,
                  'PlayingGuitar':4,'PlayingPiano':5
                  , 'PlayingSitar':6,'PlayingTabla':7,'PlayingViolin':8}
inv_video_dict = {v: k for k, v in video_dict.items()}
model_path='mobilenet_thin'
resolution = '320x240'
showBG=True
w, h = model_wh(resolution)
e = TfPoseEstimator(get_graph_path(model_path), target_size=(w, h))


# In[3]:


def process_human_data(humans):
    
    if (len(humans)==0):
        return np.zeros(shape=(18,2))
                        
    feature = np.zeros(shape=(18,2))
    for i in range(18):
        if i not in humans[0].body_parts:
            feature[i] = [0, 0]
        else:
            feature[i] = [humans[0].body_parts[i].x, humans[0].body_parts[i].y]
    
    return feature
        


# In[4]:


def check_index_range(start, end):
    if(start > end):
        raise ValueError('start index must be smaller than end index')
    elif(start < 0):
        raise ValueError('start index cannot be negative')
    elif(end>=num_class):
        raise ValueError('end index is out of range')


# In[5]:


#@profile(precision=4)  uncomment to show memory info
def inference_video(path):
    logger.debug('initialization %s : %s' % (model_path, get_graph_path(model_path)))
    cap = cv2.VideoCapture(path)

    #---------------modified----------------#
    num_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print ("All Frames: " ,num_frames)
    cur_frames = 0.0
    step = (num_frames / 20.0) 
    #---------------modified----------------#

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resize_out_ratio = 8.0
    #print("Image Size: %d x %d" % (width, height)) 

    single_video_features = np.array([])
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    while (cap.isOpened()):   
        if(cur_frames >= num_frames):
            break

        frame_no = (cur_frames/num_frames)
        cap.set(7,frame_no)
        ret_val, image = cap.read()

        #print("Frame no: ", frame_no)
        #print ("Count: ", cur_frames)

        if ret_val == True:
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
            #print ("Frame numbers: ", cur_frames, humans)
            frame_feature = process_human_data(humans) 
            single_video_features = np.append(single_video_features, frame_feature) 

        cur_frames+=step
        if cv2.waitKey(1) == 27:
            break
    #print (single_video_features)
    cv2.destroyAllWindows()
    #logger.debug('finished+')
    cap.release()
    return single_video_features
    


# In[6]:


def get_classification(classname):
    label=np.zeros(shape=(9))
    label[video_dict[classname]]=1
    return label


# In[7]:


#@profile(precision=4)  uncomment to show memory info
def create_training_data(start, end):
    check_index_range(start, end)
    mypath = Path().absolute()
    dataset_path = os.path.abspath(os.path.join(str(mypath), os.pardir))+"/action_dataset"

    feature_set=[]

    for subdir, dirs, files in os.walk(dataset_path):
        finished_dirc = 0
        for dirss in dirs:
            finished_video = 0
            if (dirss in video_dict and(video_dict[dirss]>=start and video_dict[dirss]<=end)):
                for filename in os.listdir(os.path.join(subdir,dirss)):
                    abs_path =os.path.join(subdir,dirss,filename)
                    print("inferencing video: ", filename, " from " , dirss, " directory")
                    feature =inference_video(abs_path)
                    classification = get_classification(dirss)
                    feature =list(feature)
                    feature_set.append([feature,classification])
                    print("video ", filename, " inferenced done.")
                    
                    print("# of finished videos: ", finished_video+1)
                    print("# of finished directories: ", finished_dirc)
                    finished_video+=1
                finished_dirc += 1
    pickle_file_name = 'feature_set_class_'+str(start)+'_to'+str(end)+'.pickle'
    print('writing data to pickle files.....')
    with open(pickle_file_name,'wb') as file:
        pickle.dump(feature_set,file)
    return pickle_file_name


# In[8]:


def get_video_dict():
    return video_dict


# In[9]:


def show_video_dict():
    inv_vd_dic = {v: k for k, v in video_dict.items()}
    print("ClassNumber          Class")
    print("----------------------------------")
    for i in range(len(video_dict)):
        print(i,"               ",inv_vd_dic[i])
        print("")
        

