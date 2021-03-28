# -*- coding: utf-8 -*-
"""
collection of completed functions used in the norman system
"""
#%% imports at top of the file
import cv2
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from matplotlib import pyplot as plt
from tensorflow import keras
from tkinter import *
from tkinter import filedialog
import pkg_resources
#%% extract single frame from a video #edit to allow returning of just an image
#make

def ext_frame(vid_name, frame_index, out_name=False):
    """ Extract a frame or multiple frames from a video.
    
    Keyword arguments:
        vid_name -- name of input video
        
        frame_index -- frame or frame number to be extracted. Accepts integer for single frame. Accepts tuple or list with 2 numbers representing range of frames to be extracted.
        
        out_name -- name of image file output for single frame, or folder output for multiple frames (default False). 
    
    Returns:
        If extracting single frame with out_name=False, the function returns image matrix, with a out_name it writes an image file in the current working directory. If extracting multiple frames it writes a folder of images or writes many images to the cwd. 
    
    """
    if type(frame_index) == int:
        vid = cv2.VideoCapture(vid_name)
        print("Total amount of frames: " + str(vid.get(7)))
        vid.set(1, frame_index)
        ret, frame = vid.read()
        vid.release()
        if out_name == False:
            return frame
        else:
            cv2.imwrite(out_name, frame)
    #section for mutiple frames
    else:
        vid = cv2.VideoCapture(vid_name)
        a,b = frame_index
        if out_name == False:
            out_name = ""
            slash = ""
        else:
            os.mkdir(out_name)
            slash = "\\"
        #range cant iterate tuples, but it accepts multiple integers
        for number in range(a, b):
            vid.set(1, number)
            ret, frame = vid.read()
            cv2.imwrite(out_name+slash+str(number)+".png", frame)
        vid.release()
        
#%% function to Change video resolution

# working dir must be input_dir in this case
#test with folder that only has 2 vids in

def reso_change(input_vid, out_vid, res=(640, 480)):
    """ Change resolution of a video.
    
    Arguments:
        input_vid -- name of input video file
        
        out_vid -- name you want the output file to be
        
        res -- resolution of the output file in tuple format (deafult (640,480))
    
    Returns:
        Writes a video file of the changed resolution video. Does not change original video.
    
    """
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    
    #define input vid object and videowriter object
    vid = cv2.VideoCapture(input_vid)
    fps = vid.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(out_vid, fourcc, fps, res)
        
    while(True):
        ret, frame = vid.read()
        if ret == False:
            break
        sm_frame = cv2.resize(frame, res)
        out.write(sm_frame)
    vid.release()
    out.release()

#%% add image, threshold and find contours. use TREE for hierarchy
def find_objects(image_or_path, show=False, img_out=False, im_write=False):
    """ Finds object locations images where no animal is present.
    
    Arguments:
        image_or_path -- input image variable or path to input image file
        
        show -- Boolean for whether object location result is shown in matplotlib window
        
        img_out -- Boolean for whether object location result is returned as image variable
        
        im_write -- Name of image file written if required
    
    Returns:
        Pandas dataframe with details of the objects found.
        Image variable of the objects found when img_out=True.
    """
    
    
    # if given a path to image, load image in grayscale
    if type(image_or_path) == str:
        image = cv2.imread(image_or_path, 0)
        
    # if loaded image array in colour, convert to graysclae
    elif len(np.shape(image_or_path))>2:
        image = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
    
    #if loaded image in grayscale, just use that
    else:
        image = image_or_path
    
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,11,3)
    
    thresh = cv2.medianBlur(thresh, 1)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #% Hierarchy into data frame
    hierarchy_2D = hierarchy.reshape(hierarchy.shape[1], hierarchy.shape[2])
    df = pd.DataFrame(data=hierarchy_2D, columns = ["next_cnt", "prev_cnt", "first_child", "parent"])
    #then add contour features to this data frame
    #% make centroid function
    
    def centroid(cnt):
        cnt_M = cv2.moments(cnt)
        if cnt_M["m00"] == 0:
            return (0,0)
        else:
            cx = int(cnt_M['m10']/cnt_M['m00'])
            cy = int(cnt_M['m01']/cnt_M['m00'])
            cnt_centroid = (cx, cy)
            return cnt_centroid
    
    #% make distance function and distance from centre function.
    # the centre was wrong, because shape does y first 
    img_centre = (thresh.shape[1]/2, thresh.shape[0]/2)
    def distance(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def distance_centre(point):
        img_centre = (thresh.shape[0]/2, thresh.shape[1]/2)
        return np.sqrt((img_centre[0] - point[0])**2 + (img_centre[1] - point[1])**2)
    
    #% function to merge contours into one list
            
    def merge_contours(cnt_list):
        all_cnts=[]
        for index in cnt_list:
            a = contours[index].tolist()
            all_cnts += a
        all_cnts_arr = np.asarray(all_cnts)
        return all_cnts_arr
    
    #% function for finding if 2 bounding rectangles overlap
        
    #cv2.boundingrect actually gives top left coordinates
    
    # Returns true if two rectangles overlap 
    def overlap(rect1, rect2):
        tl1 = (rect1[0], rect1[1]) #top left of rect1
        br1 = (rect1[0] + rect1[2], rect1[1] + rect1[3])#bottom right of rect1
        tl2 = (rect2[0], rect2[1]) #top left of rect2
        br2 = (rect2[0] + rect2[2], rect2[1] + rect2[3]) #bottom right of rect2 
          
        # If one rectangle is on left side of other 
        if (tl1[0] > br2[0]) or (tl2[0] > br1[0]): 
            return False
        # If one rectangle is above other 
        #needed to flip because there was of the way y is plotted
        if (tl1[1] > br2[1]) or (tl2[1] > br1[1]): 
            return False
        else:
            return True
    
    #% find centroid of every contour and add it to the data frame
    centroid_list=[]
    for index, rows in df.iterrows():
        a = centroid(contours[index])
        centroid_list.append(a)
        
    df["centroid"] = centroid_list
    df.head()
    # the (0,0) thing is affecting the average of dist, may change it to np.nan
    
    #% calculate distance from centre for every contour and add to data frame
    df["dist_from_centre"] = df["centroid"].apply(distance_centre)
    df.head()
    #% add hull area, and contour solidity
    
    hull_areas=[]
    for index, rows in df.iterrows():
        cnt_hull = cv2.convexHull(contours[index])
        hull_area = cv2.contourArea(cnt_hull)
        hull_areas.append(hull_area)
    
    df["hull_area"] = hull_areas
    df.head()
    
    
    #% add bounding rectangle.
    # top right x and y value, width and height.
    bound_rects = []
    for index in df.index:
        x = cv2.boundingRect(contours[index])
        bound_rects.append(x)
    
    df["bound_rects"] = bound_rects
    
    
    #% use pointpolygon test
    
    p2ptest_centre=[]
    for index, rows in df.iterrows():
        a = cv2.pointPolygonTest(contours[index], img_centre, True)
        p2ptest_centre.append(a)
    
    df["points2poly_test"] = p2ptest_centre
    df.head()
    
    
    
    
    #% Filter2: make new data frame using conditionals to cut contour amount down into the key objects
    
    #using y value/5 to cut out contours too far
    #add a bunch of other filters for safety
    df_cut = df[(df["points2poly_test"] > - image.shape[0]/4) & (df["hull_area"] < (image.shape[0]*image.shape[1])/20)]
    
    #% make a function to check whether bounding boxes overlap, and return a list with all overlapping
    
    def list_overlap(x):
        over_list = []
        for index, row in df_cut.iterrows():
            #if overlap is true, add to the list
            if overlap(x, row["bound_rects"]) and x != row["bound_rects"]:
                over_list.append(index)
        if over_list == []:
            return np.nan
        else:
            return over_list
    
    
    #%  use overlap function on filtered data frame
    df_cut["overlaps_w"] = df["bound_rects"].apply(list_overlap)
    # need to fix overlap function.
    #% create dataset with no nans
    df_clean = df_cut.dropna()
    #% KEY MOMENT

    obj_1 = []
    # use extend not append, because you're appending a list with a list
    # change so it starts with something that overlaps with more than one contour
    #df clean 2 only has contours which overlap with more than one object
    #df_clean = df_clean.loc[df_clean["overlaps_w"].str.len()>1]
    
    #find contour that overlaps with the most contours
    most_overlaps = df_clean.loc[df_clean["overlaps_w"].str.len()==max(df_clean["overlaps_w"].str.len())]
    
    obj_1.extend(most_overlaps.iloc[0]["overlaps_w"])
    
    for element in obj_1:
        obj_1.extend(df_clean.loc[element]["overlaps_w"])
        obj_1 = list(set(obj_1))
    
    
    # make df containing all the rows that aren't in object 1
    #the tilda ~ symbol inverts the condition
    df_clean2 = df_clean.loc[~df_clean.index.isin(obj_1)]
    #make most overlaps for object 2
    most_overlaps2 = df_clean2.loc[df_clean["overlaps_w"].str.len()==max(df_clean2["overlaps_w"].str.len())]
    
    obj_2 = []
    #get list of indexes that are not in object 1.
    # notin_obj1 = [x for x in df_clean.index.tolist() if x not in obj_1]
    # append this time because it's only one value
    obj_2.extend(most_overlaps2.iloc[0]["overlaps_w"])
    
    for element in obj_2:
        obj_2.extend(df_clean2.loc[element]["overlaps_w"])
        obj_2 = list(set(obj_2))
    
    
    #% add nans by creating class
    class cluster:
        def __init__(self, cnt_list):
            self.cnt_list = cnt_list
            self.contours = merge_contours(self.cnt_list)
            self.boundrect = cv2.boundingRect(self.contours)
            self.minrect = cv2.minAreaRect(self.contours)
            self.hull = cv2.convexHull(self.contours)
            self.centre = centroid(self.hull)
            self.left = tuple(self.hull[self.hull[:,:,0].argmin()][0])
            self.right = tuple(self.hull[self.hull[:,:,0].argmax()][0])
            self.top = tuple(self.hull[self.hull[:,:,1].argmin()][0])
            self.bottom = tuple(self.hull[self.hull[:,:,1].argmax()][0])
            self.area = cv2.contourArea(self.hull)
            if self.centre[0] < image.shape[1]/2 and self.centre[1] < image.shape[0]/2:
                self.loc = "top_left"
            if self.centre[0] < image.shape[1]/2 and self.centre[1] > image.shape[0]/2:
                self.loc = "bot_left"
            if self.centre[0] > image.shape[1]/2 and self.centre[1] < image.shape[0]/2:
                self.loc = "top_right"
            if self.centre[0] > image.shape[1]/2 and self.centre[1] > image.shape[0]/2:
                self.loc = "bot_right"
            
        def add_or_not(self):
            for index, row in df_cut.iterrows():
                if overlap(self.boundrect, row["bound_rects"]):
                    self.cnt_list.append(index)
            self.cnt_list = list(set(self.cnt_list))
                
    #% make cluster and execute add function
    obj1_cluster = cluster(obj_1)
    obj1_cluster.add_or_not()
    
    #do same for object 2
    obj2_cluster = cluster(obj_2)
    obj2_cluster.add_or_not()
    
    summary_list = [[obj1_cluster.loc, obj1_cluster.centre, obj1_cluster.hull, obj1_cluster.boundrect, obj1_cluster.minrect, obj1_cluster.area]]
    summary_list.append([obj2_cluster.loc, obj2_cluster.centre, obj2_cluster.hull, obj2_cluster.boundrect, obj2_cluster.minrect, obj2_cluster.area])
    out_df = pd.DataFrame(summary_list, columns=["location", "centre", "hull", "boundrect", "minrect", "hull_area"])
    
    

    #draw the contours on the image 
    cv2.drawContours(image, [obj1_cluster.hull], 0, (0,0,255),2) 
    cv2.drawContours(image, [obj2_cluster.hull], 0, (0,0,255),2)
    
    
    if show == True:
        plt.imshow(image)
    # return clusters
    if type(im_write) == str:
        cv2.imwrite(im_write, image)
    if img_out == True:
        return out_df, image
    else:
        return out_df

#%% convert human scoring to relavant data
    
def hs_process(hscore_file, vid_file):
    """ Converts human scoring file into labels per frame
    
    Arguments:
        hscore_file -- path to csv containing human-scored information on a video
        
        vid_file -- path to corresponding videofile that was scored
    
    Returns:
        numpy array of human labels of behaviour on a per frame basis
    """
    
    scores = pd.read_csv(hscore_file, usecols = [0,3,4], names=["time", "in_left", "in_right"], skiprows=1)
    
    #% get framerate from the video
    vid = cv2.VideoCapture(vid_file)
    fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #poses has 3240 frames
    #get frame count either from relative 
    vid.release()
    #make list times corresponding to each frame
    time_list=[timedelta(seconds=(frame_count/fps)) for frame_count in range(0, frame_total)]
    
    #% convert values from dataframe into time
    scores["format_time"] = pd.to_timedelta(scores["time"])
    
    #compare value
    min(abs(scores["format_time"]-time_list[1000]))
    
    #get location for minimun val
    a = scores.iloc[(scores["format_time"]-time_list[1000]).abs().argsort()[0]]
    a["in_left"] 
    #compare value to value in scores
    # put 0 for none, 1 for left, 2 for right
    out_list = []
    for tp in time_list:
        x = scores.iloc[(scores["format_time"]-tp).abs().argsort()[0]]
        if x["in_left"]==1:
            out_list.append(1)
        elif x["in_right"] ==1:
            out_list.append(2)
        else:
            out_list.append(0)
    #%
    out_array = np.asarray(out_list)
    return out_array

#%% make a video from directory of pictures

# this makes a video out of a directory of images
def make_video(image_folder, new_vid, fps=25):
    """ Make a video from a folder of images
    
    Arguments:
        image_folder -- folder of images to make video from
        
        new_vid -- name of new video to be made
        
        fps -- frames per second of output video (default 25)
        
    Returns:
        Writes video file.
    
    """

    slash = "\\"
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    image_files = sorted(os.listdir(image_folder))
    temp_img = cv2.imread(image_folder+slash+str(image_files[0]))
    res = (temp_img.shape[1], temp_img.shape[0])
    output_vid = cv2.VideoWriter(new_vid, fourcc, fps, res)
        
    for image in image_files:
        output_vid.write(cv2.imread(image_folder + slash + str(image)))

    output_vid.release()

#%% median filter the video to exract an image that doesn't contain the mouse
#add show conditional to make it easier to check the result, and maybe change the name of the function
    
def median_filt_video(video_name, show=False, out_file=False, select_no=15):
    """ Uses a median filter to produce a mouse-less image
    
    Arguments:
        video_name -- name of input video to be filtered
        
        show -- Boolean, determines whether the filtered image is shown as a matplotlib window (default false)
        
        out_file -- Writes an image file if given path.
        
        select_no -- Number of images used to produced filtered image (default 15)
    
    Returns:
        Image variable produced by the filtered image.

    """
    
    vid = cv2.VideoCapture(video_name)
    frameIds = vid.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=select_no)
    
    frames = []
    for fid in frameIds:
        vid.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = vid.read()
        frames.append(frame)
    
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    
    if show==True:
        plt.imshow(medianFrame)
    if type(out_file) == str:
        cv2.imwrite(out_file, medianFrame)
    return medianFrame
        

#%% the relative position function
#will have to change imports of median filter and find objects
        
#input file is the positional data csv. 
# show_fo stands for whether to show the image from the find objects execution.
def rel_pos(poses_file, objects, ret_df=False):
    """ Find relative position of mouse to both objects for every frame in a video.
    
    Arguments:
        poses_file -- path to csv file from deeplabcut tracking
        
        object -- pd data frame produced from find_objects function, containing object location and size information
        
        df -- Boolean for whether a dataframe is returned in addition to the array
        
    Returns:
        Numpy array containing relative position information per frame.
    
    
    """
#objects refers to object locations
    
    #extract df from pose data
    poses = pd.read_csv(poses_file, header=2)        
    poses.columns=["frame_no", "nose_x", "nose_y", "nose_prob", "l_ear_x", "l_ear_y", "l_ear_prob","r_ear_x", "r_ear_y","r_ear_prob", "tailbase_x", "tailbase_y", "tailbase_prob"]
    
    #% find if object is left or right object
    #hopefully find a more concise way to do this one
    if "top_left" == objects["location"][0] and "top_right" == objects["location"][1]:
        objects["l_or_r"] = ["left", "right"]
    elif "top_left" == objects["location"][1] and "top_right" == objects["location"][0]:
        objects["l_or_r"] = ["right", "left"]
    
    #top left and bottom left
    elif "top_left" == objects["location"][0] and "bot_left" == objects["location"][1]:
        objects["l_or_r"] = ["right", "left"]
    elif "top_left" == objects["location"][1] and "bot_left" == objects["location"][0]:
        objects["l_or_r"] = ["left", "right"]
    
    #top right and bottom right
    elif "top_right" == objects["location"][0] and "bot_right" == objects["location"][1]:
        objects["l_or_r"] = ["left", "right"]
    elif "top_right" == objects["location"][1] and "bot_right" == objects["location"][0]:
        objects["l_or_r"] = ["right", "left"]
    else:
        return "left_or_right_failed"
    
    #this transformation is important so that when training, 
    # the network always has the left or right object parameters separated.
    #% find average length nose to tail
    
    #import distance function
    def distance(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    poses["mouse_length"] = distance((poses["nose_x"], poses["nose_y"]), (poses["tailbase_x"], poses["tailbase_y"]))
    
    #will use the 75% quantile as the standard length
    mouse_length = poses["mouse_length"].median()
    #justify with a plot of the distribution
    #% find average mouse area
    
    def mouse_area(row):
        nose = np.array([row["nose_x"], row["nose_y"]], dtype=np.int32)
        l_ear = np.array([row["l_ear_x"], row["l_ear_y"]], dtype=np.int32)
        r_ear = np.array([row["r_ear_x"], row["r_ear_y"]], dtype=np.int32)
        tailbase = np.array([row["tailbase_x"], row["tailbase_y"]], dtype=np.int32)
        polygon = np.array([nose, l_ear, r_ear, tailbase], dtype=np.int32)
        row["mouse_area"] = cv2.contourArea(polygon)
        return row
        
    poses = poses.apply(mouse_area, axis=1)
    
    #use 75% quantile as the standaerd area
    mouse_area = poses["mouse_area"].median()
    #% calculate distance from mouse body parts to centre
    
    #making function which inputs pose and object data frames, and output's distances to centre
    
    def dist2centres(pos_df, obj_df ):
        #could incroporate finding mouse length into the function
        #making tuples to use as coordinates
        nose = (pos_df["nose_x"], pos_df["nose_y"])
        l_ear = (pos_df["l_ear_x"], pos_df["l_ear_y"])
        r_ear = (pos_df["r_ear_x"], pos_df["r_ear_y"])
        tailbase = (pos_df["tailbase_x"], pos_df["tailbase_y"])
        
        #making variables for centres of left and right objects for readability
        centre_l = objects["centre"][objects["l_or_r"]=="left"].iloc[0]
        centre_r = objects["centre"][objects["l_or_r"]=="right"].iloc[0]
        
        #initialising output dataframe
        out = pd.DataFrame((distance(nose, centre_l)/mouse_length)/5, columns=["nose2centre_l"])
        
        #adding distances to centre of object1
        out["l_ear2centre_l"] = (distance(l_ear, centre_l)/mouse_length)/5
        out["r_ear2centre_l"] = (distance(r_ear, centre_l)/mouse_length)/5
        out["tailbase2centre_l"] = (distance(tailbase, centre_l)/mouse_length)/5
        
        #adding distance to centre of object 2
        out["nose2centre_r"] = (distance(nose, centre_r)/mouse_length)/5
        out["l_ear2centre_r"] = (distance(l_ear, centre_r)/mouse_length)/5
        out["r_ear2centre_r"] = (distance(r_ear, centre_r)/mouse_length)/5
        out["tailbase2centre_r"] = (distance(tailbase, centre_r)/mouse_length)/5
        
        # now normalise these values further using multiples of mouse lenght
        # normalise to 5 mouse lengths. If dist is over 5 mouse lengths,
        # then set value to 1 as it's so far away
        for mycol in out.columns:
            out.loc[(out[mycol] > 1) , mycol] = 1 
        
        return out
    
    #successful
    
    #% calculate distance to edge of object using points to polygon test
    
    def dist2objects(pos_df, obj_df):
        
        #divide everything by 5 in order to scale these values
        
        #get hull
        hull_l = objects["hull"][objects["l_or_r"]=="left"].iloc[0]
        hull_r = objects["hull"][objects["l_or_r"]=="right"].iloc[0]
        
        #get all body points, have to reformat because cv2.pointpolygontest can't take in series
        nose = []
        for a, b in zip(pos_df["nose_x"], pos_df["nose_y"]):
            nose.append((a,b))
        nose_dist=[]
        for coord in nose:
            nose_dist.append(cv2.pointPolygonTest(hull_l, coord, True)/mouse_length)
        out = pd.DataFrame(nose_dist, columns=["nose2obj_l"])
            
        l_ear = []
        for a, b in zip(pos_df["l_ear_x"], pos_df["l_ear_y"]):
            l_ear.append((a,b))
        l_ear_dist=[]
        for coord in nose:
            l_ear_dist.append(cv2.pointPolygonTest(hull_l, coord, True)/mouse_length)
        #add to dataframe
        out["l_ear2obj_l"] = l_ear_dist
            
        r_ear = []
        for a, b in zip(pos_df["r_ear_x"], pos_df["r_ear_y"]):
            r_ear.append((a,b))
        r_ear_dist=[]
        for coord in nose:
            r_ear_dist.append(cv2.pointPolygonTest(hull_l, coord, True)/mouse_length)
        #add to dataframe
        out["r_ear2obj_l"] = r_ear_dist
            
        tailbase = []
        for a, b in zip(pos_df["tailbase_x"], pos_df["tailbase_y"]):
            tailbase.append((a,b))
        tailbase_dist=[]
        for coord in nose:
            tailbase_dist.append(cv2.pointPolygonTest(hull_l, coord, True)/mouse_length)
        #add to dataframe
        out["tailbase2obj_l"] = tailbase_dist
    
        ####### for object 2
        nose = []
        for a, b in zip(pos_df["nose_x"], pos_df["nose_y"]):
            nose.append((a,b))
        nose_dist=[]
        for coord in nose:
            nose_dist.append(cv2.pointPolygonTest(hull_r, coord, True)/mouse_length)
        out["nose2obj_r"] = nose_dist
            
        l_ear = []
        for a, b in zip(pos_df["l_ear_x"], pos_df["l_ear_y"]):
            l_ear.append((a,b))
        l_ear_dist=[]
        for coord in nose:
            l_ear_dist.append(cv2.pointPolygonTest(hull_r, coord, True)/mouse_length)
        #add to dataframe
        out["l_ear2obj_r"] = l_ear_dist
            
        r_ear = []
        for a, b in zip(pos_df["r_ear_x"], pos_df["r_ear_y"]):
            r_ear.append((a,b))
        r_ear_dist=[]
        for coord in nose:
            r_ear_dist.append(cv2.pointPolygonTest(hull_r, coord, True)/mouse_length)
        #add to dataframe
        out["r_ear2obj_r"] = r_ear_dist
            
        tailbase = []
        for a, b in zip(pos_df["tailbase_x"], pos_df["tailbase_y"]):
            tailbase.append((a,b))
        tailbase_dist=[]
        for coord in nose:
            tailbase_dist.append(cv2.pointPolygonTest(hull_r, coord, True)/mouse_length)
        #add to dataframe
        out["tailbase2obj_r"] = tailbase_dist
        
        # normalise the positive values (within the shape of the test objects) to 0
        # actually dont, it may provide important information
        for mycol in out.columns:
            out[mycol] = out[mycol]/5
        #    out.loc[(out[mycol] > 0) , mycol] = 0  #for possible use
    
        return out
    #% calculate angle betwen mouse and centre
    
    #make midpoint function
    def midpoint(point1, point2):
        return ((point2[0] - point1[0])/2 , (point2[1] - point1[1])/2)
    
    #1 create triangle between nose and centre of an object 
    #using dx, dy, and distance nose to object centre to get triangle parameters. 
    
    def find_angles(pos_df, obj_df):
        nose = (pos_df["nose_x"], pos_df["nose_y"])
        l_ear = (pos_df["l_ear_x"], pos_df["l_ear_y"])
        r_ear = (pos_df["r_ear_x"], pos_df["r_ear_y"])
        tailbase = (pos_df["tailbase_x"], pos_df["tailbase_y"])
        
        #making variables for centres of objects for readability
        centre_l = objects["centre"][objects["l_or_r"]=="left"].iloc[0]
        centre_r = objects["centre"][objects["l_or_r"]=="right"].iloc[0]
        
        #making (right angled) triangle of nose to centre of object
        dx = abs(nose[0]-centre_l[0])
        dy = abs(nose[1]-centre_l[1])
        hyp = distance(nose, centre_l)
        
        #calculate tan angle using opposite/adjacent. calc reverse of it using 90 degree corner rule
        theta1_obj_l = 90 - np.rad2deg(np.arctan(dy/dx))
        # convert into sin and cos angle. angle from nose to left object
        sin_obj_l = np.sin(theta1_obj_l)
        cos_obj_l = np.cos(theta1_obj_l)
        
        #create output dataframe with sin angle
        out = pd.DataFrame(sin_obj_l, columns= ["nose2objl_sine"])
        # add cos angle to dataframe
        out["nose2objl_cos"] = cos_obj_l
        
        # now calculate the outside angle of nose to mid point
        #(true north then rotate clockwise to from nose to midpoint of ears)
        ear_mid = midpoint(l_ear, r_ear)
        dx = abs(nose[0]-ear_mid[0])
        dy = abs(nose[1]-ear_mid[1])
        hyp = distance(nose, ear_mid)
        
        #180 subtract angle because we're finding the outside angle in relation to true north
        theta2 = 180 - np.rad2deg(np.arctan(dx/dy))
        #convert to sin and cos angle. For angle nose to midpoint
        sin_n2m = np.sin(theta2)
        cos_n2m = np.cos(theta2)
        
        #add to data frame
        out["nose2mid_sine"] = sin_n2m
        out["nose2mid_cos"] = cos_n2m
        
        #get outside angle from midpoint between ears to the tail
        #make right-angled triangle
        dx = abs(tailbase[0]-ear_mid[0])
        dy = abs(tailbase[1]-ear_mid[1])
        hyp = distance(ear_mid, tailbase)
        #180 - because we're finding the outside angle
        theta3 = 180 - np.rad2deg(np.arctan(dx/dy))
        #convert to sin and cos angle. For midpoint to the tail
        sin_m2t = np.sin(theta3)
        cos_m2t = np.cos(theta3)
        
        #add to data frame
        out["mid2tail_sine"] = sin_m2t
        out["mid2tail_cos"] = cos_m2t
        
        #making (right angled) triangle with second object
        dx = abs(nose[0]-centre_r[0])
        dy = abs(nose[1]-centre_r[1])
        hyp = distance(nose, centre_r)
        
        #calculate tan angle using opposite/adjacent. calc reverse of it using 90 degree corner rule
        theta1_obj_r = 90 - np.rad2deg(np.arctan(dy/dx))
        #convert to sin and cos. angle from nose to the right object 
        sin_obj_r = np.sin(theta1_obj_r)
        cos_obj_r = np.cos(theta1_obj_r)
        
        #add to data frame
        out["nose2objr_sine"] = sin_obj_r
        out["nose2objr_cos"] = cos_obj_r
        
        return out
        
    
    #% merge all outputs into one dataframe
    
    rel_pos = pd.concat([find_angles(poses, objects), dist2centres(poses, objects), dist2objects(poses, objects)], axis=1)
    
    rel_pos_array = rel_pos.to_numpy(dtype=float)
    
    if ret_df == False:
        return rel_pos_array
    else:
        return rel_pos_array, rel_pos

#%% visualisation of norman labelling mouse behaviour as left, right or neither
         

def draw_vid(y_labels, input_vid, out_name=False):
    """ Makes a video with left, right and neither labels overlayed.
    
    Arguments:
        y_labels -- array containing a label of attention for each frame in the video
        
        input_vid -- input video to be overlayed
        
        out_name -- name of video created. If false will add "norman" to the existing file name
    
    Returns:
        Writes video file
    """
    #put in the input vid
    in_vid = cv2.VideoCapture(input_vid)
    
    #create conditional so you can make your own file name or change it
    if out_name==False:    
        out_file = input_vid[:-4]+"_norman.mp4"
    else:
        out_file = out_name
        
    #get fps from video to use in output vid
    fps = in_vid.get(cv2.CAP_PROP_FPS)
    #get resolution from video to use in output vid
    h = int(in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(in_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    #make the output vid
    #define the codec
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(out_file, fourcc, fps, (w,h))
    
    #test if using a loop based off the array of labels works
    for label in y_labels:
        #extract frame
        ret, frame = in_vid.read()
        #make sure it is read correctly
        if ret == True:
            #add writing to frame dependant on the label
            #if statement for corresponding text to show for each label
            # and for colour to show.
            if label == 0:
                text = "neither"
                colour = (255, 255, 255)
            if label == 1:
                text = "left"
                colour = (255,0, 0)
            if label == 2:
                text = "right"
                colour = (0,0,255)
            #get shape of frame to find place to the image
            fr_y = frame.shape[0]
            fr_x = frame.shape[1]
            # select font
            font = cv2.FONT_HERSHEY_PLAIN
            # add text to the frame
            cv2.putText(frame, text, (int(fr_x*0.2), int(fr_y*0.9)), font, 4, colour, 4, cv2.LINE_AA)
            
            #write the frame to the video
            out.write(frame)
        else:
            break
          
    #release the input and output videos
    in_vid.release()
    out.release()
    return "complete"

#%% function to label a video using norman

def label_vid(relative_pos, model_path="normal"):
    """ Uses norman model to predict classes using relative position
    
    Arguments:
        model_path -- path to the model file of norman. The defaults argument loads the default packaged norman model.
        
        relative_pos -- array of relative position information on a per-frame basis
    
    Returns:
        Array of labels porduced by the norman network on each frame.
    """
    #determine whether to load default package norman model, or another
    if model_path=="normal":
        model_path = pkg_resources.resource_filename("norman_ai", "data/norman_model1.h5")
    
    #load the norman network model from a file
    norman_model = keras.models.load_model(model_path)
    
    #use norman model from file to interpret relative position as classes
    labels = norman_model.predict_classes(relative_pos)
    
    #return the labels per frame
    return labels

#%% function to calculate the discrimination index
    
# it requires the video file to get the fps, 
# the labels for frames spent interacting with the left or right object,
# and the location of the novel object in the video ("left" or "right")
#later could include conditional for type of equation used to calculate DI
def calc_di(video, labels, no_loc, ret_all = True):
    """ Calculates the discrimination index for a video.
    
    Arguments:
        video -- path to video file
        
        labels -- norman labels for each frame in the video
        
        no_loc -- location of the novel object, can be "left" or "right" respective to the mouse starting position
        
    Returns:
        Discrimination index, time spent with left object, time spent with right object, and fps of the video.
    
    """

    # load the video as a cv2 object
    vid = cv2.VideoCapture(video)
    # divide 1 by fps of the video, this will help us see how much time the interacting frames sum up to
    fps_ratio = 1/(vid.get(cv2.CAP_PROP_FPS))
    vid.release()
    # get the amount of time spent interacting with the left object and right object
    time_left = (labels == 1).sum()*fps_ratio
    time_right = (labels == 2).sum()*fps_ratio
    
    #the equation for discrimination index depends on which object is novel
    if no_loc == "left":
        di = (time_left - time_right)/(time_left + time_right)
    if no_loc == "right":
        di = (time_right - time_left)/(time_left + time_right)
    
    #return the discrimination index and the amount of time spent with the left and right objects
    if ret_all == True:
        return di, time_left, time_right, 1/fps_ratio
    else:
        return di
        
#%% create "norkid" class. It will be used to hold data from a video and pose file
    
class norkid:
    """ A class used to hold all neccesary information about a video
    
    Attributes:
        video_name -- the name of the input video as a string
        
        median_img -- a frame from the input video without the mouse
        
        object_locs -- object locations; the output of the find_objects funtion
        
        fo_img -- image of find_objects function result
        
        relative_pos -- relative position of mouse to objects in array
        
        labels -- frame labels produced by norman predictions
        
        di -- discrimination index for the video
        
        tl -- seconds mouse spent with left object
        
        tr -- seconds mouse spent with right object
        
        fps -- fps of input video
    
    Methods:
        draw_vid -- makes a video with an overlay of norman predictions
    
    """
    #the class takes in a video and the associated deeplabcut pose file for it
    def __init__(self, video, pose_file, no_loc, model_path="normal"):
        #have the name of the video as a string
        self.video_name = video
        #the median image is a frame from the video without the mouse 
        self.median_img = median_filt_video(video)
        # object locations are the output of the find objects funtion
        self.object_locs, self.fo_img = find_objects(self.median_img, img_out = True)
        # use relative position
        self.relative_pos = rel_pos(pose_file, self.object_locs)
        #get labels produced by the classification by norman
        self.labels = label_vid(self.relative_pos, model_path)  
        #get the discrimination index, seconds spent with left or right, and fps
        self.di, self.tl, self.tr, self.fps = calc_di(video, self.labels, no_loc)      

#adding the draw vid function to this class
    def draw_vid(self, out_name=False):
        
    #put in the input vid
        video = self.video_name
        in_vid = cv2.VideoCapture(video)
        #input the lables
        y_labels = self.labels
        
        #create conditional so you can make your own file name or change it
        if out_name==False:    
            out_file = video[:-4]+"_norman.mp4"
        else:
            out_file = out_name
            
        #get fps from video to use in output vid
        fps = in_vid.get(cv2.CAP_PROP_FPS)
        #get resolution from video to use in output vid
        h = int(in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(in_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        #make the output vid
        #define the codec
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(out_file, fourcc, fps, (w,h))
        
        #test if using a loop based off the array of labels works
        for label in y_labels:
            #extract frame
            ret, frame = in_vid.read()
            #make sure it is read correctly
            if ret == True:
                #add writing to frame dependant on the label
                #if statement for corresponding text to show for each label
                # and for colour to show.
                if label == 0:
                    text = "neither"
                    colour = (255, 255, 255)
                if label == 1:
                    text = "left"
                    colour = (255,0, 0)
                if label == 2:
                    text = "right"
                    colour = (0,0,255)
                #get shape of frame to find place to the image
                fr_y = frame.shape[0]
                fr_x = frame.shape[1]
                # select font
                font = cv2.FONT_HERSHEY_PLAIN
                # add text to the frame
                cv2.putText(frame, text, (int(fr_x*0.2), int(fr_y*0.9)), font, 4, colour, 4, cv2.LINE_AA)
                
                #write the frame to the video
                out.write(frame)
            else:
                break
              
        #release the input and output videos
        in_vid.release()
        out.release()
        cv2.destroyAllWindows()
        return "complete"

#%% the gui function

def run_gui():
    """ Launches graphical user interface to analyse a video using the norman system """
    #make a window
    # the tk function starts the main window of the application
    window = Tk()
    #add a title
    window.title("NORMAN system")
    #define the size of the window
    window.geometry("600x800")
    
    #make a welcome message
    # it appears label comes out as long horizontal text
    label1 = Label(text = "Welcome to the NORMAN system", font = "Helvetica 9 bold")
    label1.place(x=180, y = 25)
    
    #initialise path variables
    
    def select_video():
        #global so the variable is set outside of the function also
        global video_path
        video_path = filedialog.askopenfilename(initialdir = "C:\"")
        video_box["text"] = video_path
    
    #make function for selecting pose file 
    def select_pose():
        global pose_path
        pose_path = filedialog.askopenfilename(initialdir = "C:\"")
        pose_box["text"] = pose_path
        
    #make function for selecting machine learning model
    def select_model():
        global ml_path
        ml_path = filedialog.askopenfilename(initialdir = "C:\"")
        model_box["text"] = ml_path
        
    #function to get what was typed into the novel box
    def run_analysis():
        global no_location
        no_location = entry_box.get()
        global norkid1
        #use the gathered information to make a norkid object
        try:
            ml_path
        except NameError:
            ml_path_exists = False
        else:
            ml_path_exists = True
        #instructing to use default model
        if ml_path_exists == False:
            ml_path = "normal"
        norkid1 = norkid(video_path, pose_path, no_location, ml_path)
        
        #need to do the output boxes within the function
        di_out["text"] = str(round(norkid1.di, 5)) + " au" 
        tl_out["text"] = str(round(norkid1.tl, 5)) + " seconds"
        tr_out["text"] = str(round(norkid1.tr, 5)) + " seconds"
        
    
    
    #make a button that will select a video file
    video_button =  Button(window, text = "Please select a video", command = select_video)
    video_button["bg"] = "light green"
    video_button.place(x = 50, y =75, width = 200, height=25)
    
    #this button will select a pose file 
    pose_button =  Button(window, text = "Please select a pose file", command = select_pose)
    pose_button["bg"] = "light green"
    pose_button.place(x = 300, y =75, width = 200, height=25)
    
    #this button will start running norman
    run_button = Button(window, text = "Accept and run video analysis", command = run_analysis)
    run_button["bg"] = "light green"
    run_button.place(x= 125, y = 500, width = 300, height = 25)
    
    model_button = Button(window, text = "Please select the NORMAN model file.", command = select_model)
    model_button["bg"] = "light green"
    model_button.place(x = 125, y =335, width = 300, height = 25)
    
    #add display output boxes to show which files are loaded
    #this will be for the video file
    # it appears Message comes as multiple lines of text
    video_box = Message(text = "")
    video_box["relief"] = "sunken"
    video_box["bg"] = "white"
    video_box.place(x= 50, y = 125, width = 200, height = 100)
    
    #the box is for the pose file
    pose_box = Message(text = "")
    pose_box["relief"] = "sunken"
    pose_box["bg"] = "white"
    pose_box.place(x= 300, y = 125, width = 200, height = 100)
    
    #this display is for the norman model file
    model_box = Message(text = "")
    model_box["relief"] = "sunken"
    model_box["bg"] = "white"
    model_box.place(x = 180, y = 375, width = 200, height = 100)
    
    #label asking which object is novel
    label2 = Label(text = "Which object in the video is novel? Please enter left or right.", font="Helvetica 9 bold")
    label2.place(x = 100, y = 245)
    
    
    
    #drop down box for "left or right"
    #list_box = Listbox(window, selectmode=SINGLE)
    #list_box.insert(1, "left")
    #list_box.insert(2, "right")
    #list_box.place(x= 225, y=225, width = 100, height = 50)
    
    #make entry box to type left or right
    entry_box = Entry()
    entry_box.place(x= 225, y=275, width = 100, height = 30)
    #entry_box.focus()
    
    #get the output of the DI, time with left, time with right, into an output box
    di_label = Label(text = "Discrimination index", font = "Helvetica 9 bold")
    di_label.place(x = 50, y = 550)
    
    #make output box for di
    di_out = Label(text = "")
    di_out["relief"] = "sunken"
    di_out["bg"] = "white"
    di_out.place(x = 50, y = 575, width = 100, height = 25)
    
    #title for left
    tl_label = Label(text = "Time interacting with left object")
    tl_label.place(x = 50, y = 625)
    
    #make output box for time interacting with left
    tl_out = Label(text = "")
    tl_out["relief"] = "sunken"
    tl_out["bg"] = "white"
    tl_out.place(x = 50, y = 650, width = 125, height = 25)
    
    #title for right
    tr_label = Label(text = "Time interacting with right object")
    tr_label.place(x = 50, y = 700)
    
    #make output box for time interacting with right
    tr_out = Label(text = "")
    tr_out["relief"] = "sunken"
    tr_out["bg"] = "white"
    tr_out.place(x = 50, y = 725, width = 125, height = 25)
    
    #need a button to display the find objects image, and a button for creating a labelled nort video
    
    #function to display find objects result
    def display_fo():
        plt.imshow(norkid1.fo_img)
        
    #function to make labelled video
    def make_label_vid():
        norkid1.draw_vid()
    
    #button to display find objects
    fo_button = Button(window, text = "Display object detection results", command = display_fo)
    fo_button["bg"] = "light blue"
    fo_button.place(x = 300, y = 550, width = 275, height = 50)
    
    # button for drawing video
    draw_button = Button(window, text = "Visualisation of NORMAN labelling. \nCreates mp4 video.", command = make_label_vid)
    draw_button["bg"] = "orange"
    draw_button.place(x = 300, y = 625, width = 275 , height = 50) 
    
    #this loop keeps the program working apparently
    window.mainloop()



















