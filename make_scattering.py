# import packages
from kymatio import Scattering2D
import kymatio

import torch.nn as nn
import torch.optim
import torch
import torch.utils.data as utils

import time
import sys

import numpy as np
from numpy.lib.format import open_memmap

import cv2

#=========================================================================================================
# parse input
train_name = sys.argv[1]
ng = int(sys.argv[2])


#=========================================================================================================
# add shape noise
def add_shape_noise(x, ng):

    # parameter
    sige=0.4
    scale = 60*3.5
    map_size = 512
    A = (float(scale)/map_size)**2

    """Add shape noise"""
    sigpix = sige / (2 * A * ng)**0.5  # final pixel noise scatter

    # add shape noise to map
    return x + np.random.normal(loc=0, scale=sigpix, size=x.shape)

#------------------------------------------------------------------------------------
# smooth data
def smooth(x):

    # map size in arcmin
    map_size_arcmin = 60*3.5

    # smooth scale
    smoothing_scale_arcmin = 1.

    """Smooth by Gaussian kernel."""
    # smoothing kernel width in pixels instead of arcmins
    map_size_pix = x.shape[0]
    s = (smoothing_scale_arcmin * map_size_pix) / map_size_arcmin

    # cut off at: 6 sigma + 1 pixel
    # for large smooothing area and odd pixel number
    cutoff = 6 * int(s+1) + 1

    # return map
    return cv2.GaussianBlur(x, ksize=(cutoff, cutoff), sigmaX=s, sigmaY=s)


#=========================================================================================================
# main body of the script
def main():

    # load data
    training_x = np.load('../weak_lensing_data/sparse_grid_final_512pix_x_' + train_name + '.npy')[:,:,:,0]
    print(training_x.shape)

    # add noise
    training_x = add_shape_noise(training_x, ng)

    # smooth the image
    for i in range(training_x.shape[0]):
        training_x[i,:,:] = smooth(training_x[i,:,:])
        if i%1e3 == 0:
            print(i)

#----------------------------------------------------------------------------------------------------------
    # define scattering
    scattering = Scattering2D(J=5, shape=(training_x[0,:,:].shape), L=2, max_order=3)
    scattering.cuda()

    # initiate results array
    Sx = []

#----------------------------------------------------------------------------------------------------------
    # loop over batches of 500 objects
    for i in range(training_x.shape[0]//100+1):
        print(i)

        # record time
        start_time = time.time()

        # transform to torch tensors
        tensor_training_x = torch.from_numpy(training_x[100*i:100*(i+1),:,:]).type(torch.cuda.FloatTensor)

        # perform scattering
        Sx.append(scattering(tensor_training_x).mean(dim=(2,3)).cpu().detach().numpy())
        print(time.time() - start_time)

#----------------------------------------------------------------------------------------------------------
    # save results
    for i in range(len(Sx)):
        try:
            Sx_array = np.vstack([Sx_array,Sx[i]])
        except:
            Sx_array = Sx[i]
    print(Sx_array.shape)
    np.save("Sx_" + train_name + "ing_expected_" + str(ng) + ".npy", Sx_array)

if __name__ == '__main__':
    main()
