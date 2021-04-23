# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:44:24 2021

@author: seyij
"""


# import neccessary functions
import norman_ai.norman_functions as nf
import cv2
from matplotlib import pyplot as plt
#%% create an analysed norman object

ex = nf.norkid("test_79.mp4", "test_79_poses_filtered.csv", "norman_model1.h5", "left")

#%% plot and save an image using cv2 or matplotlib

plt.imshow(ex.median_img)

plt.imsave("med.png", ex.median_img)

plt.imsave("outline.png", ex.fo_img)

cv2.imwrite("outline.png", ex.fo_img)

#%% get help on a function

help(nf.make_video)

#%% Extract frame example


# Extract a single frame to produce an image array
image_array = nf.ext_frame("test_79.mp4", 42)

# Extract a frame and save as an image file
nf.ext_frame("test_79.mp4", 42, out_name = "my_frame.png")

# Extract frames 42 to 150 to the current working directory
nf.ext_frame("test_79.mp4", (42, 150))

# Extract frames 42 to 150 to a new folder
nf.ext_frame("test_79.mp4", (42, 5000), out_name = "many_frames")

#%% Resolution change example
import norman_functions as nf

#change resolution of video to default value of 640x480
nf.reso_change("test_79.mp4", "new_res.mp4")

#change resolution of video to a new value of 640x480
nf.reso_change("test_79.mp4", "new_res.mp4", res=(789,456))

#%% Find objects example

#use median filter to generate a mouse-free image array or file 
box_image = nf.median_filt_video("test_79.mp4")
nf.median_filt_video("test_79.mp4", out_file="box_img.png")

#use the object detector on an image array, show the result, save the output as pandas dataframe
x = nf.find_objects(box_image, show=True)
#use object detector on an image file, show the result, save output as pandas dataframe
x = nf.find_objects("box_img.png", show=True)

#use object detector and output resulting image array with dataframe
x, image = nf.find_objects(box_image, img_out=True)

#use object detector and write the output image as a file
x = nf.find_objects(box_image, im_write="outlines.png")

#%% Make video example

# make a new video from a folder of images with an fps of 25
nf.make_video("many_frames", "vid_new.mp4")

# make a new video from a folder of images with any fps
nf.make_video("many_frames", "vid_new.mp4", fps=60)

#%% Median filter example

#show what a single frame looks like before filtering
plt.imshow(nf.ext_frame("test_79.mp4", 42))

#use median filter with default 15 frames, show the result, output to variable and to file
x = nf.median_filt_video("test_79.mp4", show=True, out_file="filtered.png")

#use median filter with specified number of frames
x = nf.median_filt_video("test_79.mp4", select_no=20)

#%% Draw vid example

#input video name
video = "test_79.mp4"
#input path to norman model
model_path = "norman_model1.h5"
#input path to pose file 
pose_file = "test_79_poses_filtered.csv"

#extract the median image 
median_img = nf.median_filt_video(video)
# extract object locations 
object_locs = nf.find_objects(median_img)
# calculate relative position of mouse to objects
relative_pos = nf.rel_pos(pose_file, object_locs)
#get labels produced by the prediction by norman
labels = nf.label_vid(model_path, relative_pos)  

#make norman labelled video, with default naming of the original video name +_norman
nf.draw_vid(labels, video)

#make norman labelled video, with specified name
nf.draw_vid(labels, video, out_name="labelled_vid.mp4")

#for a label vid example take the first part of the draw vid example.

#%% calc di example

#input video name
video = "test_79.mp4"

#extract labels from video
model_path = "norman_model1.h5"
pose_file = "test_79_poses_filtered.csv"
median_img = nf.median_filt_video(video) 
object_locs = nf.find_objects(median_img)
relative_pos = nf.rel_pos(pose_file, object_locs)
labels = nf.label_vid(model_path, relative_pos) 

# calculate and store just the di of a video.
# we are saying the novel object is the left one
di = nf.calc_di(video, labels, "left", ret_all = False)

# store the di, time with left, time with right and fps (default)
#we are saying the novel object is the right one
di, tl, tr, fps = nf.calc_di(video, labels, "right")

#show results rounded to 2 decimal places
print(round(di, 2), round(tl , 2), round(tr,2) , round(fps, 2))
#%% norkid object example

#get video name, norman model file and deeplabcut tracking file
video = "test_79.mp4"
model_path = "norman_model1.h5"
pose_file = "test_79_poses_filtered.csv"

#make a norkid object
x = nf.norkid(video, pose_file, model_path, "left")

# view name of input video
x.video_name
#show the median filtered image
plt.imshow(x.median_img)
# show find objects outline image
plt.imshow(x.fo_img)

#view object location datafrane
x.object_locs

#view relative position dataframe
x.relative_pos

#view norman labels per frame
x.labels

#show the DI, time with left, time with right, and the video fps
print(round(x.di, 2), round(x.tl , 2), round(x.tr,2) , round(x.fps, 2))

# make a norman labelled video with a default name (video name + _norman)
x.draw_vid()
# make a norman labelled video with a specified name
x.draw_vid(out_name="anything.mp4")


#%% help code example

# create a list variable
list1 = [3,5,4,53,5,35,3]
#ask what a list variable is
help(list1)
#ask what type of variable this variable is
type(list1)

#norman example
import norman_functions as nf

# ask for documentation for the function
help(nf.find_objects)

#make a norkid object
video = "test_79.mp4"
model_path = "norman_model1.h5"
pose_file = "test_79_poses_filtered.csv"
x = nf.norkid(video, pose_file, model_path, "left")

#ask for details about this class 
help(x)
# ask what type x is
type(x)

#%%

import deeplabcut as dlc

dlc.launch_dlc()

#%% Vid to di command line example

#get video name
video = "test_79.mp4"

#import deeplabcut
import deeplabcut as dlc

#get the project config file location
# the r makes the string be treated as a raw string
#you will have to chan
path_cfg = r"C:\Users\seyij\Documents\packaging_norman\demo\nort_demo-Jesusanmi-2019-09-26\config.yaml"

#analyse the video and save the results to a csv
dlc.analyze_videos(path_cfg, video, videotype="mp4", save_as_csv=(True))

#filter the predcitions from video analysis for smoother tracking
dlc.filterpredictions(path_cfg, video, save_as_csv=(True))

#make a deeplabcut labelled video
dlc.create_labeled_video(path_cfg, [video], videotype='.mp4', filtered=True, save_frames=False)

#get path to pose document
poses = r"test_79DLC_resnet_50_nort_demoSep26shuffle1_50000_filtered.csv"

#create norkid python object
# stating the novel object is the left object in the experiment
x = nf.norkid(video, poses, "left")

#show the discrimination index, the time spent with left object, time spent with right object
print("DI:"+str(x.di)+", Time left:"+str(x.tl)+ ", Time right:"+ str(x.tr))

#make an video visualisation of norman labelling the mouse video
#this is optional as not every video result will need visualisation
x.draw_vid()





















