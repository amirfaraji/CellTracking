import os
from glob import glob
import random
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

PATCH_SIZE = 256

def pad_image(img):
  ##### Pad to PATCH_SIZE divisible dimensions #####
  if np.mod(img.shape[0], 2) != 0:
    img = img[:-1,:]

  if np.mod(img.shape[1], 2) != 0:
    img = img[:,:-1]

  row_rem = np.mod(img.shape[0], PATCH_SIZE)
  col_rem = np.mod(img.shape[1], PATCH_SIZE)

  if row_rem < (PATCH_SIZE // 2):
    row_pad = (PATCH_SIZE // 2) - row_rem
  else:
    row_pad = PATCH_SIZE - row_rem

  if col_rem < (PATCH_SIZE // 2):
    col_pad = (PATCH_SIZE // 2) - col_rem
  else:
    col_pad = PATCH_SIZE - col_rem

  padded_img = np.pad(img, ((row_pad//2,row_pad//2), (col_pad//2,col_pad//2)), 'edge')

  return padded_img, [row_rem, row_pad, col_rem, col_pad]

def create_patches(pad_img, img_windows):
  ##### Create patches #####
  if np.mod(pad_img.shape[0], PATCH_SIZE) == 0:
    num_rows = pad_img.shape[0] // PATCH_SIZE * 2 - 1
  else:
    num_rows = pad_img.shape[0] // PATCH_SIZE * 2
  if np.mod(pad_img.shape[1], PATCH_SIZE) == 0:
    num_cols = pad_img.shape[1] // PATCH_SIZE * 2 - 1
  else:
    num_cols = pad_img.shape[1] // PATCH_SIZE * 2

  counter = 0

  for row in np.arange(num_rows):
    for col in np.arange(num_cols):
      cT = row*(PATCH_SIZE // 2)
      cL = col*(PATCH_SIZE // 2)
      
      window_img = pad_img[cT:cT+PATCH_SIZE, cL:cL+PATCH_SIZE]
      
      img_windows.append(window_img)

      counter += 1

  return img_windows

def fetch_imgsmsks(msk_name_list, in_dir):
  img_patches = []
  msk_patches = []
  for msk_name in msk_name_list:
    img_name = os.path.join(in_dir, 't' + msk_name[-8:])
    img, msk = imread(img_name), imread(msk_name)
    msk[msk > 0] = 1
    
    img_padded, _ = pad_image(img)
    msk_padded, _ = pad_image(msk)
    
    img_patches = create_patches(img_padded, img_patches)
    msk_patches = create_patches(msk_padded, msk_patches)
    
  img_patches = np.array(img_patches).reshape(-1, PATCH_SIZE, PATCH_SIZE, 1)
  msk_patches = np.array(msk_patches).reshape(-1, PATCH_SIZE, PATCH_SIZE, 1)

  return img_patches, msk_patches

def fetch_imgs(img_name_list):
  img_patches = []
  for img_name in img_name_list:
    img = imread(img_name)
    
    img_padded, pad_vals = pad_image(img)
    
    img_patches = create_patches(img_padded, img_patches)
    
  img_patches = np.array(img_patches).reshape(-1, PATCH_SIZE, PATCH_SIZE, 1)

  return img_patches, pad_vals

def load_train_data(in_dir):
    ### Choose sequence and get images ###
    msk_names = glob(os.path.join(in_dir + '_GT', 'SEG/') + '*.tif')
    random.Random(12).shuffle(msk_names)

    trn_msk_names = msk_names[:round(0.70*len(msk_names))]
    val_msk_names = msk_names[round(0.70*len(msk_names)):round(0.85*len(msk_names))]
    tst_msk_names = msk_names[round(0.85*len(msk_names)):]

    train_imgs, train_masks = fetch_imgsmsks(trn_msk_names, in_dir)
    val_imgs, val_masks = fetch_imgsmsks(val_msk_names, in_dir)
    test_imgs, test_masks = fetch_imgsmsks(tst_msk_names, in_dir)
    print('==> Train Images: ' + train_imgs.shape, ', Train Masks: ' +  train_masks.shape)
    print('==> Validation Images: ' + val_imgs.shape, ', Validation Masks: ' +  val_masks.shape)
    print('==> Test Images: ' + test_imgs.shape, ', Test Masks: ' +   test_masks.shape)

    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks

def load_test_data(in_dir):
    img_names = glob(os.path.join(in_dir) + '*.tif')
    imgs, pad_vals = fetch_imgs(img_names)
    print('==> Images: ' + imgs.shape)

    return imgs, pad_vals

