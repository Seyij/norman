# The NORMAN System

The Novel Object Recognition Mouse Analysis Network (NORMAN) system is a package for automation of the novel object recognition behavioural assay in neuroscience studies.

PyPi link: https://pypi.org/project/norman-ai/

This is part of work that will be presented as a poster at the [BNA Festival 2021](https://meetings.bna.org.uk/bna2021/) which runs from the 12-15th of April. It was developed as part of an MSci degree at the University of Dundee. The package and repository will be updated with video outlines and other details shortly, in the run up to the BNA festival.

Author: Oluwaseyi Jesusanmi  
Twitter: @neuroseyience   


![NORMAN labelling example0](https://github.com/Seyij/norman/blob/master/media/slide11_vid_Trim.gif)
![NORMAN labelling example1](https://github.com/Seyij/norman/blob/master/media/test_79_norman_Trim.gif)

---
### Contents
1. [System Overview](#system-overview)
2. [System Requirements](#system-requirements)
3. [Installation guide](#installation-guide)
4. [Video to DI walkthrough](#video-to-di-walkthrough)
5. [Function details](#function-details)
6. [Using Deeplabcut with NORMAN](#using-deeplabcut-with-norman)
7. [Glossary](#glossary)



### System Overview

The novel object recognition test (NORT) compares the time a mouse spends with a new/novel object and a familiar object in order to assess a mouses memory. The quantitative output of the test is reffered to as the discrimination index (DI), and is what many behavioural neuroscience studies use. It is calculated using (tN - tF)/(tN + tF), where tN is time spent with novel object and tF is time spent with familiar object.

The NORMAN system can automatically extract the DI and other metrics from a video of a NORT. The general steps in functioning of the system are as follows:

1. Use a deeplabcut network to track a mouse's position across a NORT video.
2. Automatically find object locations/spatial information.
3. Calculate relative positions of mouse to each object for every video frame.
4. Use an artificial neural network that I built and trained to understand mouse-object interaction (NORMAN) to label each frame with the relevant interaction.
5. Extract metrics, and visualise NORMAN's labelling.


The following guide uses [Anaconda](https://www.anaconda.com/products/individual) and conda for environment mangement, with python for programming. The program used for environment management was anaconda prompt, for quick python access I used ipython within anaconda prompt, and for in depth script-writing and development I used spyder 4. Other programs are available.

The guide assumes a very basic familiarity with anaconda/conda and python, but a user interface is available for those not comfortable coding. If help is needed feel free to contact me, I'd be happy discuss, make changes, and/or post more tutorials. I have also included a glossary in this document for some terms used.

### System requirements

Operating System - Windows 10. This is due to using the [Tensorflow-directml](https://github.com/microsoft/tensorflow-directml) machine learning API. Tensorflow-directml has simple installation, and allows tensorflow use with hardware accelleration on ANY Directx12 capable GPU. That includes AMD, Nvidia, and even Intel GPUs! The only negative is that it is still under active development, but I will make compatibility changes as neccessary.

Hardware - Tested with laptop i7-9750h, 16Gbs of Ram, RTX 2060 mobile with 6Gbs of VRAM. You must have a GPU for the system to work. On the described hardware, a 5 minute real-time novel object recognition video will be processed in 15 seconds by NORMAN system, with the deeplabcut tracking step bringing total processing time to 2 minutes 30 seconds. I reccommend any directx12 capable GPU with at least 6Gbs of VRAM.

### Installation guide

By the end of this section you should have an environment that works with deeplabcut, NORMAN, and hardware accelleration through tensorflow-directml. You can manage conda environments using the click-through user interface in Anaconda Navigator, or you can use the command line in anaconda prompt. This guide I will be using the anaconda prompt command line interface. For conda environment management commands, refer to this [conda cheat sheet](https://kapeli.com/cheat_sheets/Conda.docset/Contents/Resources/Documents/index).

```bash
#Create initial environment, ensuring the correct version of h5py is installed
conda create -n norman_dlc python=3.6 h5py=2.8.0
# Activate the environment so you can install more packages on it
conda activate norman_dlc
# install tensorflow with a directml backend
pip install tensorflow-directml
# install the animal tracking library
pip install deeplabcut
# install user interface library for deeplabcut.
pip install -U wxPython==4.0.7.post2
#install spyder for general coding (optional step, not required)
conda install spyder
#install functioning version of plotting library
conda install matplotlib=3.1.3
#install norman
pip install norman-ai
```
Please note the version of matplotlib may need to be updated depending on the order of installation. Now that installation is complete, perform tests to see if installation was successful using ipython. Ipython is a python program useful for working at the command line.

```bash
#open the ipython program
ipython
```
```python
#check if your graphics card is recognised by direcml after installation
#within ipython program import tensorflow
import tensorflow as tf
#type code to check if gpu device is recognised
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#check if norman was installed
import norman_ai.norman_functions as nf
nf.run_gui()
#a user interface should appear
```

### Video to DI walkthrough

This section will take you through the basic steps required to get a DI from a video and to visualise NORMAN'S labelling, both using the provided user interface and with code. Make sure you are in the "norman_dlc" environment before proceeding.

##### __*User interface method*__

This uses the video file, a premade deeplabcut tracking file and an option norman model file in the demo folder of the project repository.

```bash
#open the ipython program
ipython
```
```python
#import norman
import norman_ai.norman_functions as nf
#open graphical user interface (gui)
nf.run_gui()
```
1. Input the video file "test_79.mp4" by clicking on "please select a video", a file selection window will open.
![gui1](https://github.com/Seyij/norman/blob/master/media/gui1_m.png)
2. Select the corresponding deeplabcut tracking csv "test_79_poses_filtered.csv" file by clicking on "please select a pose file".
![gui2](https://github.com/Seyij/norman/blob/master/media/gui2_m.png)
3. Enter which object is the video is novel to the mouse.
![gui3](https://github.com/Seyij/norman/blob/master/media/gui3_m.png)
4. Click accept and run video analysis. (Selecting the norman model is optional).
5. Results will be in the bottom left.
![gui4](https://github.com/Seyij/norman/blob/master/media/gui4_m.png)
6. To show whether the find_objects function worked correctly, press the "Display object detection" button.
![gui5](https://github.com/Seyij/norman/blob/master/media/gui5.png)
7. Press the "Visualisation of NORMAN labelling" button to produce a video showing how NORMAN labelled the video. The video will be in the same directory as the original video, with "norman" added to the name.
* Play the video and see how NORMAN does.



##### __*Code-based method*__
To follow the code walkthrough set the working directory to the demo folder in the project repository.

```python
#import norman functions
import norman_ai.norman_functions as nf
#import deeplabcut functions
import deeplabcut as dlc

#get video name
video = "test_79.mp4"

```
Deeplabcut needs a path to a deeplabcut project config file in order for its functions to work. A pretrained network is provided in the demo folder. Please note if using a full path to the config file, it may be different to the paths in the tutorial depending on where you store the repository.

```python

#get the deeplabcut project config file location.
#the r makes the string be treated as a raw string
path_cfg = r"\nort_demo-Jesusanmi-2019-09-26\config.yaml"

#analyse the video and save the results to a csv
dlc.analyze_videos(path_cfg, video, videotype="mp4", save_as_csv=(True))

#filter the predcitions from video analysis for smoother tracking
dlc.filterpredictions(path_cfg, video, save_as_csv=(True))
```
From here NORMAN functions use information from the video file combined with the csv file for analysis.

```python

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

```
If you have trouble with this demo, please see the function details, which contain code examples and guidance on other functions in the NORMAN package. When attempting this with your own data, the most common error is related to the find_objects() function, as not all NORT mouse chambers are compatible with the object detection algorithm. If this is the case with your data I am happy to work on variant object detectors.

### Function details


Though very few NORMAN functions are needed to extract a discrimination index from a video, the NORMAN system has more functions for working with NORT images and videos. In this section I will explain how to use various functions in the library, with their use-cases and code examples. Where relevant I will also include information on how/why I made the function for context. Code examples below were done in the ???demo??? folder of the NORMAN project repository, using the ???test_79.mp4??? video as the source data. The top of each code example with show the arguments of the function and their defaults when applicable. For the full code of each function see the norman_functions.py file, and for the help doc strings use the python default function help().

```python
#norman help example
import norman_ai.norman_functions as nf
# ask for documentation for the function
help(nf.find_objects)

```

##### __*Norkid python class, object parameters and methods*__  
The "norkid" class handles all the processes to analyse a video, while holding the data for a video in one place. To create a norkid object you need a path to a video, a tracking csv, and which experiment object is the novel object (left or right). A norkid object stores: an image with the mouse filtered out, the find object output, the relative position function output, the labels produced by the NORMAN model, the DI and related metrics. These properties can be used in conjunction with other functions and libraries, while easing the debugging process.

```python
#%% norkid object example
import norman_ai.norman_functions as nf
from matplotlib import pyplot as plt

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

```
![norkid1](https://github.com/Seyij/norman/blob/master/media/norkid1_med2_fo1.png)

```python
# show find objects outline image
plt.imshow(x.fo_img)
```
![norkid2](https://github.com/Seyij/norman/blob/master/media/norkid2_fo2.png)

```python
#view object location datafrane
x.object_locs

#view relative position dataframe
x.relative_pos

#view norman labels per frame
x.labels

#show the DI, seconds spent with left and right objects, and the video fps
print(round(x.di, 2), round(x.tl , 2), round(x.tr,2) , round(x.fps, 2))
-0.36 20.75 43.83 18.07

# make a norman labelled video with a default name (video name + _norman)
x.draw_vid()
# make a norman labelled video with a specified name
x.draw_vid(out_name="my_norman_vid.mp4")

```

##### __*Frame extractor ??? ext_frame()*__  
Extract a frame or multiple frames from a video. Outputs an image from a video as an image array, a file, or a folder image files. Often when working with video data, specific frames may need to be checked or analysed, as many functions work on a frame-by-frame basis.

```python
# Function arguments = ext_frame(vid_name, frame_index, out_name=False)
# Extract frame example 
import norman_ai.norman_functions as nf
# set video name
my_video = ???test_79.mp4???

# Extract a single frame to produce an image array
image_array = nf.ext_frame(my_video, 42)

# Extract a frame and save as an image file
nf.ext_frame(my_video , 42, out_name = "my_frame.png")

# Extract frames 42 to 150 to the current working directory
nf.ext_frame(my_video,  (42, 150))

# Extract frames 42 to 150 to a new folder
nf.ext_frame(my_video, (42, 150), out_name = "many_frames")

```
##### __*Resolution changer ??? reso_change()*__
Changes the resolution of a video to a specified value on the x and y axis. Creates a new video, does not affect the data of the original video.

```python
# Function Arguments: reso_change(input_vid, out_vid, res=(640, 480))
# Resolution change example.
#Please note new video is created, original video is not affected
import norman_ai.norman_functions as nf
# set video name
my_video = ???test_79.mp4???

#change resolution of video to default value of 640x480
nf.reso_change(my_video, "new_res.mp4")

#change resolution of video to a new arbitrary value
nf.reso_change(my_video, "new_res.mp4", res=(789,456))
```

##### __*Moving object remover/filter ??? median_filt_video()*__
This function removes the mouse from the video and produces an image of the bare experiment box and objects. This must be done before using the find_objects() function for object detection, as the mouse will interfere with the object finder. It will remove any moving object from a video given enough frames.
```python
#Function arguments: median_filt_video(video_name, show=False, out_file=False, select_no=15)
#%% Median filter example
import norman_ai.norman_functions as nf
from matplotlib import pyplot as plt
# set video name
my_video = ???test_79.mp4???

#show what a single frame looks like before filtering
plt.imshow(nf.ext_frame(my_video, 42))
```
![med1](https://github.com/Seyij/norman/blob/master/media/med1.png)

```python
#use median filter with default 15 frames, show the result, output to variable and to file
x = nf.median_filt_video(my_video, show=True, out_file="filtered.png")
```
![med2](https://github.com/Seyij/norman/blob/master/media/norkid1_med2_fo1.png)

```python
#use median filter with specified number of frames, output to variable
x = nf.median_filt_video(my_video, select_no=20)
```


##### __*Object localiser ??? find_objects()*__
This automatically finds the 2 experiment objects and outputs the locations, area and outline of each of them. The values are used later to calculate proximity of the mouse to objects. This function should be used after the median filter has been used to create a mouse-free image. So far in testing it reliably finds the location of objects on videos where the floor of the test chamber is plain. An alternative version of the function is under exploration for boxes with patterned floors or large amounts of debris, as this affects the edge detection algorithms.

```python
# Function Arguments: find_objects(image_or_path, show=False, img_out=False, im_write=False)
# Find objects example
import norman_ai.norman_functions as nf
from matplotlib import pyplot as plt
# set video name
my_video = ???test_79.mp4???

#use median filter to generate a mouse-free image array or file 
box_image = nf.median_filt_video(my_video)
nf.median_filt_video(my_video, out_file="box_img.png")

#show image using matplotlib. Note it may show in false colour
plt.imshow(box_image)
```
![fo1](https://github.com/Seyij/norman/blob/master/media/norkid1_med2_fo1.png)
```python
#use the object detector on an image array, show the result, save the output as pandas dataframe
x = nf.find_objects(box_image, show=True)
```
![fo2](https://github.com/Seyij/norman/blob/master/media/norkid2_fo2.png)

```python
#use object detector on an image file, show the result, save output as pandas dataframe
x = nf.find_objects("box_img.png", show=True)

#use object detector and output resulting image array with dataframe
x, image = nf.find_objects(box_image, img_out=True)

#use object detector and write the output image as a file
x = nf.find_objects(box_image, im_write="outlines.png")
```

##### __*Video maker from multiple images ??? make_video()*__
This function takes in a directory of images, and outputs a video created from these images at any specified framerate. 

```python
#Function arguments: make_video(image_folder, new_vid, fps=25)
# Make video example
import norman_ai.norman_functions as nf

# make a new video from a folder of images with an fps of 25
nf.make_video("many_frames", "vid_new.mp4")

# make a new video from a folder of images with any fps
nf.make_video("many_frames", "vid_new.mp4", fps=60)
```
##### __*Making predictions from position data ??? label_vid()*__
Prediction in the case of the NORMAN system refers to the NORMAN network model taking relative positional information per-frame, then determining what the mouse is paying attention to in each frame. In the output array, neither=0, left object =1, right object=2. These labels are used for calculating the DI or other useful metrics. Since the labels are per-frame, many NORT attention metrics can be calculated, including latency to first interaction and change in interaction amount over the time course of an experiment.

```python
#Function arguments: label_vid(model_path, relative_pos)
# Label vid example
import norman_ai.norman_functions as nf

#input video name
video = "test_79.mp4"
#input path to pose file 
pose_file = "test_79_poses_filtered.csv"

#extract the median image 
median_img = nf.median_filt_video(video)
# extract object locations 
object_locs = nf.find_objects(median_img)
# calculate relative position of mouse to objects
relative_pos = nf.rel_pos(pose_file, object_locs)
#get labels produced by the prediction by norman
labels = nf.label_vid(relative_pos)  

```

##### __*Making a NORMAN labelled video ??? draw_vid()*__
The draw_vid() function visualises the NORMAN model predictions (see figure). This is useful for assessing whether the system is working correctly as it can be judged by eye. It is also useful when presenting data to show how the NORMAN system works.

```python
# Function arguments: draw_vid(y_labels, input_vid, out_name=False)
# Draw vid example
import norman_ai.norman_functions as nf

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

#make norman labelled video, with default naming of the original video name +_norman saved to the current working directory
nf.draw_vid(labels, video)

#make norman labelled video, with specified name saved to specified directory.
nf.draw_vid(labels, video, out_name="labelled_vid.mp4")

```
##### __*Calculating discrimination index ??? calc_di()*__
This function only produced the DI, time spent with each object and the fps.

```python
# calc di example
import norman_ai.norman_functions as nf

#input video name
video = "test_79.mp4"

#extract labels from video
pose_file = "test_79_poses_filtered.csv"
median_img = nf.median_filt_video(video) 
object_locs = nf.find_objects(median_img)
relative_pos = nf.rel_pos(pose_file, object_locs)
labels = nf.label_vid(relative_pos) 

# calculate and store just the di of a video.
# we are saying the novel object is the left one
di = nf.calc_di(video, labels, "left", ret_all = False)
# store the di, time with left, time with right and fps (default)
#we are saying the novel object is the right one
di, tl, tr, fps = nf.calc_di(video, labels, "right")

#show results rounded to 2 decimal places
print(round(di, 2), round(tl , 2), round(tr,2) , round(fps, 2))
0.35 20.81 43.66 18.07

```

### Using Deeplabcut with NORMAN.

Deeplabcut is a python library for marker-less tracking of animals using artificial neural networks. It provides a variety of functions centred around streamlining the process of tracking animals. The general workflow of the library is to use a neural network that has been pretrained on animal movement, train it to recognise features on the animal you wish to track by labelling images, then use the trained network to track animal movement on a selection of videos. For in depth details please refer to the official [deeplabcut tutorials on github](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/UseOverviewGuide.md). Here I will briefly discuss points that are relevant for using Deeplabcut in tandem with the NORMAN system.

Installation ??? The tutorials recommend tensorflow-gpu installation for using Deeplabcut with hardware accelleration. But the previously discussed tensorflow-directml based environment allows a lot easier installation when using a GPU and allows you to have much more customizable environments, while remaining compatible with deeplabcut.

Config file ??? When you start a new deeplabcut project it creates a folder with a ???config.yaml??? file. This config file is what all the functions use to find other files in the directory, such as the location of the trained models. The entire project folder can be moved, copied and shared. As long as the information in the config file is up do date, all the functions should work as normal.

Training a deeplabcut model ??? When labelling frames to train Deeplabcut to track mice or rats, please ensure the labels are added in the right order with the correct text. This is to ensure the output csv from deeplabcut analysis is compatible with the NORMAN system. The list of labels is: nose, l_ear, r_ear, tailbase. L_ear refers to left ear, r_ear refers the right ear, and tailbase refers to where the tail connects to the main body of the mouse. If you are training a Deeplabcut network with a GPU that has less than 8Gbs of video memory (VRAM), you may run into memory allocation errors. To solve this you must change the session parameters to allow a procedural change in memory allocation during training . The following code implements this:

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
```
The Deeplabcut model zoo ??? Deeplabcut has a growing ???model zoo???, where you can download pre-trained models to use for animal tracking. These are very convenient as the most time consuming step (training and evaluating the model), is eliminated. There is no pretrained mouse tracking model available yet, but when one is released, this should be used and NORMAN will be given a function to accept the csv files produced when using this model.


Video resolution - The processing time/resolution trade off must also be considered. Higher resolution can mean higher accuracy as there is more detail visible, but higher resolutions can greatly increase the processing time for the deeplabcut tracking step. The point of diminishing returns from resolution differs depending on the application. For example if you are tracking the nose and ears of a mouse, a very low resolution of just 480p would provide good tracking accuracy. Whereas if you were tracking whiskers at that same resolution with the mouse equidistant from the camera, the whiskers will be indistinguishable and the accuracy would suffer tremendously.

### Glossary

Array ??? A data type for storing sequences of a variety of dimensions, usually in the numpy library. It can be used to store a simple sequence of numbers, or something more complex like an image by representing each pixel colour with a number in a table-like structure. Array processing is incredibly fast, making them useful for matrices calculations.

Class ??? In python, a class refers to a type of object and is used as a layout on which to build an object. An object in python has properties (parameters and characteristics), and methods (types of functions) associated with it. For an imaginary example, say a pet golden retriever was a python object. It would have properties such as height, weight, breed, fur colour etc. It could have methods such as run, play fetch, eat, sleep etc. These methods could also affect some of the object properties, for example the ???eat??? method may increase the dog object???s weight. The class for this example would be a dog class. It would be a common set of properties and methods of which the exact values of which will be different for each new dog object that is made. Python is an object orientated programming language, meaning that many data and variable are often stored as objects/object parameters, with methods which can be applied to them. For a practical implementation of a class and object, see the norkid class explanation in section 4.

Computer vision ??? A field of computer science programming that aims to extract information from images and video. This often works by converting images into an array of numbers which represent pixels, and using algebraic functions on these numbers to extract the necessary information. Recently much of computer vision research utilises artificial neural network techniques as they can apply incredibly complex mathematical functions to images automatically, if given enough training data.
Frame/video frame ??? Videos are made up of many images in sequence, each individual image is called a frame. Most standard video is shot at about 30 frames per second, meaning each second of video is made out of 30 images.

GPU/integrated GPU ??? A graphical processing unit is a specialised computer component made for processing highly parallel mathematical problems. Originally produced for computer graphics and video-games, it was discovered that the parallel nature of the hardware is well suited to running artificial neural network operations. GPUs are often parts of graphics cards, while integrated GPUs are less powerful devices that are integrated into CPU chipsets, usually for devices without dedicated graphics cards.

Library/python library ??? Sets of reusable code. A library often includes a series of functions that are helpful for a specific type of programming problem. For example pandas is a tabular data processing library, which includes sets of functions useful for processing tabular data.

Object ??? See entry for ???Class???.

Package ??? Package is a collection of code similar to a library. It is referred to as a package when it is released in a distributable form that can be installed into an environment.

Version control ??? A method of archiving previous version of a coding project, commonly the git package is used to do this. Using this method all files and all changes to files are stored. This means that if the code breaks, you can roll-back to a previous version where the code was working. It is also important for scientific reproducibility, since all changes are tracked, any mistakes or instances of data being changed in a fraudulent manner can be tracked.

Virtual development environment ??? A partition on a computer where many libraries and packages can be installed, in a way that other aspects of the computer are not affected. This allows developers to work on multiple different projects that require different packages without running into as many compatibility errors.

