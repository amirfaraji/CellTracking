import keras.backend as K
import tensorflow as tf
import numpy as np

def jaccard_index(y_true, y_pred, smooth=1e-12):
    """Jaccard Index"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jaccard_index_loss(y_true, y_pred):
    """Jaccard Loss Function"""
    return 1-jaccard_index(y_true, y_pred)

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def weighted_jaccard_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    numer = K.sum(K.minimum(y_true_f, y_pred_f)) + K.epsilon()
    denom = K.sum(K.maximum(y_true_f, y_pred_f)) + K.epsilon()
    loss = 1 - numer/denom
    return loss

def tversky_loss(y_true, y_pred, beta=0.2):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + \
                 (1 - beta) * y_true * (1 - y_pred)

    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

def binary_focal_loss(y_true, y_pred):
    """
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred) 

    gamma = 2. 
    alpha = .25

    pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
    pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))

    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, K.epsilon(), 1. - K.epsilon())
    pt_0 = K.clip(pt_0, K.epsilon(), 1. - K.epsilon())

    FL1 = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
    FL0 = -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return FL1 + FL0

def dice_coeff(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    denom = K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + K.epsilon()

    return (2. * intersection + K.epsilon()) / denom
