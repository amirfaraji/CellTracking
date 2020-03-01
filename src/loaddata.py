import os
from glob import glob
import random
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

in_dir = 'drive/My Drive/CellTracking/BF-C2DL-MuSC'
PATCH_SIZE = 256

def pad_image(img, msk):
  ##### Pad to PATCH_SIZE divisible dimensions #####
  if np.mod(img.shape[0], 2) != 0:
    img = img[:-1,:]
    msk = msk[:-1,:]

  if np.mod(img.shape[1], 2) != 0:
    img = img[:,:-1]
    msk = msk[:,:-1]

  # print(img.shape)

  # plt.imshow(img)
  # plt.title('Original Image')
  # plt.show()

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

  # print(row_rem, row_pad, col_rem, col_pad)

  padded_img = np.pad(img, ((row_pad//2,row_pad//2), (col_pad//2,col_pad//2)), 'edge')
  padded_msk = np.pad(msk, ((row_pad//2,row_pad//2), (col_pad//2,col_pad//2)), 'edge')

  # print(padded_img.shape)

  return padded_img, padded_msk

def create_patches(pad_img, pad_msk, img_windows, msk_windows):
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
      
      window_msk = pad_msk[cT:cT+PATCH_SIZE, cL:cL+PATCH_SIZE]
      window_img = pad_img[cT:cT+PATCH_SIZE, cL:cL+PATCH_SIZE]
      
      msk_windows.append(window_msk)
      img_windows.append(window_img)

      counter += 1

  return img_windows, msk_windows

def fetch_imgs(msk_name_list):
  img_patches = []
  msk_patches = []
  for msk_name in msk_name_list:
    img_name = os.path.join(in_dir, folder[:2], 't' + msk_name[-8:])
    img, msk = imread(img_name), imread(msk_name)
    msk[msk > 0] = 1
    
    img_padded, msk_padded = pad_image(img, msk)
    
    img_patches, msk_patches = create_patches(img_padded, msk_padded, img_patches, msk_patches)
    
  img_patches = np.array(img_patches).reshape(-1, PATCH_SIZE, PATCH_SIZE, 1)
  msk_patches = np.array(msk_patches).reshape(-1, PATCH_SIZE, PATCH_SIZE, 1)

  return img_patches, msk_patches

def load_data(in_dir):
  ### Choose sequence and get images ###
  for folder in os.listdir(in_dir):
    if not folder.endswith('1_GT'):
      continue

    msk_names = glob(os.path.join(in_dir, folder, 'SEG/') + '*.tif')
    random.Random(12).shuffle(msk_names)

    trn_msk_names = msk_names[:round(0.70*len(msk_names))]
    val_msk_names = msk_names[round(0.70*len(msk_names)):round(0.85*len(msk_names))]
    tst_msk_names = msk_names[round(0.85*len(msk_names)):]

    train_imgs, train_masks = fetch_imgs(trn_msk_names)
    val_imgs, val_masks = fetch_imgs(val_msk_names)
    test_imgs, test_masks = fetch_imgs(tst_msk_names)
    print('==> Train Images: ' + train_imgs.shape, ', Train Masks: ' +  train_masks.shape)
    print('==> Validation Images: ' + val_imgs.shape, ', Validation Masks: ' +  val_masks.shape)
    print('==> Test Images: ' + test_imgs.shape, ', Test Masks: ' +   test_masks.shape)
