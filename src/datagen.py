import numpy as np
from keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator

class DataGen(Sequence):
  def __init__(self, imgs, masks, weights, batch_size, shuffle):
    self.imgs = imgs
    self.masks = masks
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.aug = ImageDataGenerator(
      horizontal_flip=True, 
      vertical_flip=True,
      shear_range=0.05, 
      rotation_range=20,
      width_shift_range=0.1, 
      height_shift_range=0.1
    )
    self.weights = weights
  
  def on_epoch_end(self):
    if self.shuffle:
      np.random.RandomState(seed=444).shuffle(self.imgs)
      np.random.RandomState(seed=444).shuffle(self.masks)
      np.random.RandomState(seed=444).shuffle(self.weights)

  def __len__(self):
    return self.imgs.shape[0] // self.batch_size

  def __getitem__(self,idx):
    img_batch = self.imgs[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
    mask_batch = self.masks[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
    sample_weights = self.weights[idx*self.batch_size:(idx+1)*self.batch_size]

    if self.shuffle:
      transform_dict = {
        'theta'          : (np.random.rand()*20.0),
        'tx'             : (np.random.rand()/10.0),
        'ty'             : (np.random.rand()/10.0),
        'shear'          : (np.random.rand()/20.0),
        'flip_horizontal': bool(round(np.random.rand())),
        'flip_vertical'  : bool(round(np.random.rand()))
      }

      img_batch = np.array([self.aug.apply_transform(img, transform_dict) \
          for img in img_batch])
      mask_batch = np.array([self.aug.apply_transform(mask, transform_dict) \
          for mask in mask_batch])

    return img_batch, mask_batch, sample_weights
