import sys
import glob
import h5py
import random
import argparse
import skimage.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from evaluation.metrics import jaccard_index
from models.networks import unet
from processing.postprocessing import crf_seg

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split


##################################
###      Argument parser       ###
##################################
parser = argparse.ArgumentParser(description='Inference File for Cell Tracking ')

parser.add_argument('image_path', type=str, help='A required string argument for path to image')
parser.add_argument('weight_path', type=str, help='A required string argument for path to weight')

parser.add_argument('--crf_flag', type=bool, help='An optional boolean crf flag argument', default=True)

args = parser.parse_args()


##################################
###   Load Image of Dataset    ###
##################################
# Load Image of Dataset
hdf5_dir = args.image_path 
hdf5file = h5py.File(hdf5_dir + "/seg_samples.hdf5", "r")
test_set = np.array(hdf5file["/images"]).astype("uint8")
hdf5file.close()


##################################
###   Load Model and Predict   ###
##################################
opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model = unet((256,256), 16)
model.load_weights(args.weight_path)
model.compile(opt, loss=jaccard_index)

pred = model.predict(test_set)

if (args.crf_flag):
    for i in range(len(test_set)):
        pred = crf_seg(test_set[i,:,:], pred[i,:,:])