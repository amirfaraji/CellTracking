# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from evaluation.metrics import *
from models.networks import *
from loaddata import *
from datagen import DataGen

from keras.optimizers import Adam

RANDOM_SEED = 42
PATCH_SIZE  = 256
BATCH_SIZE  = 16
IMG_CHN     = 1


input_dir = "../Data/BF-C2DL-MuSC/01"
save_dir = "../src/weights/"
train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = \
    load_train_data(input_dir.split('/')[-2], input_dir.split('/')[-1])

opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

mdl = unet((PATCH_SIZE, PATCH_SIZE, IMG_CHN), 16)
mdl.compile(
    opt,
    loss=weighted_jaccard_loss,
    metrics=[jaccard_index, precision_m, f1_m, dice_coeff]
)

init_weights = np.ones((train_imgs.shape[0]))
train_gene = DataGen(train_imgs, train_masks, init_weights, BATCH_SIZE, True)

weights = np.ones(train_imgs.shape[0])
  
def jacc_loss(y_true, y_pred):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  numer = np.mean(np.minimum(y_true_f, y_pred_f)) + 1e-7
  denom = np.mean(np.maximum(y_true_f, y_pred_f)) + 1e-7
  return 1 - numer/denom


lr_counter = 0
lr_val = 1e-4
lr_min = 1e-7
es_counter = 0
loss_values = []
loss_values_val = []
while es_counter < 30:
    if lr_counter == 15 and lr_val > lr_min:
        lr_val = np.max([lr_val*0.3, lr_min])
        K.set_value(mdl.optimizer.lr, lr_val)
        lr_counter = 0
    
    results = mdl.fit_generator(
        train_gene,
        epochs=1,
        validation_data=(val_imgs, val_masks),
        steps_per_epoch=len(train_imgs) // BATCH_SIZE,
    )
    
    if not loss_values_val or \
        results.history['val_loss'] < np.array(loss_values_val).min():
        
        lr_counter = 0
        es_counter = 0
        mdl.save_weights(
            save_dir + 'BF-C2DL-MuSC.hdf5', 
            overwrite=True
        )
    else:
        lr_counter += 1
        es_counter += 1

    loss_values.append(results.history['loss'])
    loss_values_val.append(results.history['val_loss'])

    train_gene = DataGen(train_imgs, train_masks, weights=init_weights, \
                          batch_size=BATCH_SIZE, shuffle=False)
    
    train_pred = mdl.predict_generator(train_gene)

    for p_ind,pred in enumerate(train_pred):
      weights[p_ind] = jacc_loss(pred, train_masks[p_ind])

    train_gene = DataGen(train_imgs, train_masks, weights=weights, \
                         batch_size=BATCH_SIZE, shuffle=True)


