#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os


# In[2]:


def load_tree_attributes(inputPath):
    # load the CSV file using Pandas
    df = pd.read_csv(inputPath)
    # return the data frame
    return df


# In[6]:


def load_tree_images(df, inputPath):
    # initialize our images array (i.e., the house images themselves)
    images = []
    # loop over the indexes of the houses
    for i in df.ID.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, "{}.tif".format(i)])
        treePaths = sorted(list(glob.glob(basePath)))
        # loop over the input house paths
        for treePath in treePaths:
            # load the input image, resize it to be 224 224, and then
            # update the list of input images
            image = cv2.imread(treePath)
            image = cv2.resize(image, (124,124))
            images.append(image)
    # return our set of images
    return np.array(images)


# In[ ]:




