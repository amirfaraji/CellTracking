# -*- coding: utf-8 -*-
import sys
import glob
import h5py
import random
import skimage.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from evaluation.metrics import *
from models.networks import *

from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TEST_SPLIT  = 0.15
VALID_SPLIT = 0.2
PATCH_SIZE  = 256
BATCH_SIZE  = 16
IMG_HEIGHT  = 1036 
IMG_WIDTH   = 1070 
IMG_CHN     = 1


hdf5_dir = "/content/drive/My Drive/CellTracking" # sys.argv[1] #
hdf5file = h5py.File(hdf5_dir + "/seg_samples.hdf5", "r")
img_patches = np.array(hdf5file["/images"]).astype("uint8")
msk_patches = np.expand_dims(np.array(hdf5file["/masks"]).astype("uint8"), 3)
img_patches = np.stack((img_patches,)*3, axis=3)
msk_patches[msk_patches > 0] = 1
hdf5file.close()



train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    img_patches, msk_patches, test_size=TEST_SPLIT, random_state=RANDOM_SEED
)

train_imgs, valid_imgs, train_masks, valid_masks = train_test_split(
    train_imgs, train_masks, test_size=VALID_SPLIT, random_state=RANDOM_SEED
)

opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

mdl = unet_VGG16((PATCH_SIZE,PATCH_SIZE,IMG_CHN, 3))
mdl.summary()
mdl.compile(
    loss=weighted_jaccard_loss,
    optimizer=opt, 
    metrics=[jaccard_index, precision_m, f1_m, dice_coeff]
  )

checkpoint = ModelCheckpoint("weights2.hdf5", monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='min')

earlystopping = EarlyStopping(monitor='val_loss', verbose = 1,
                              min_delta = 1E-5, patience = 30, mode='min')

callbacks_list = [checkpoint, earlystopping]

aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, 
                         shear_range=0.05, rotation_range=20, 
                         width_shift_range=0.1, height_shift_range=0.1)

def data_gene(BATCH_SIZE):
  img_datagen = aug.flow(train_imgs, batch_size=BATCH_SIZE, seed=444)
  msk_datagen = aug.flow(train_masks, batch_size=BATCH_SIZE, seed=444)

  for (img,mask) in zip(img_datagen, msk_datagen):
        yield (img,mask)

results = mdl.fit_generator(data_gene(BATCH_SIZE),
                  epochs=150, 
                  validation_data=(valid_imgs, valid_masks),
                  steps_per_epoch=len(train_imgs) // BATCH_SIZE,
                  callbacks=callbacks_list)
