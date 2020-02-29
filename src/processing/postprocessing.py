import numpy as np
import h5py
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_labels


def crf_seg(img, mask, gt_prob=0.9):
    labels = mask.flatten() 

    M = 2  # number of labels

    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    # Certainty that the ground truth is correct
    GT_PROB = gt_prob

    U = unary_from_labels(labels, M, GT_PROB, False)
    d.setUnaryEnergy(U)

    feats = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=np.expand_dims(img, 2), chdim=2)

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)

    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    return res
