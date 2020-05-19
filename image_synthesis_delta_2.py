# import packages
from kymatio import Scattering2D
import kymatio

import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data as utils

import time

import numpy as np


#=========================================================================================================
# main body of the script
def image_synthesis(image,
                    learnable_param_list = [(100, 1e-3)],
                    savedir = '/home/yting',):

    image = image[None, :, :]
    image_GPU = torch.from_numpy(image).type(torch.cuda.FloatTensor) + 5

#----------------------------------------------------------------------------------------------------------
    # define mock image
    class model_image(nn.Module):
        def __init__(self):
            super(model_image, self).__init__()

            # initialize with white noise GRF
            if True:
                self.param = torch.nn.Parameter(torch.randn(1,256,256).type(torch.cuda.FloatTensor)*0.2+6)

#---------------------------------------------------------------------------------------------------------

    model_fit = model_image()

    # define learnable
    for learnable_group in range(len(learnable_param_list)):
        num_step = learnable_param_list[learnable_group][0]
        learning_rate = learnable_param_list[learnable_group][1]

        # optimizer = optim.Adam(model_fit.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model_fit.parameters(), lr=learning_rate)

        # optimize
        for i in range(int(num_step)):

            # loss: L2
            loss_L2 = ((((model_fit.param.reshape(1,256,256) - 5)**2).mean()**0.5 - \
                       ((image_GPU-5)**2).mean()**0.5) / ((image_GPU-5)**2).mean()**0.5 )**2

            # loss: L1
            loss_L1 = (( (model_fit.param.reshape(1,256,256) - 5).abs().mean() - (image_GPU-5).abs().mean() )\
                        /(image_GPU-5).abs().mean() )**2

            # loss: mean
            loss_mean = ((model_fit.param.reshape(1,256,256) - 5).mean() - (image_GPU-5).mean())**2

            loss = loss_L1 + loss_L2*0 #+ loss_mean

            if i%100== 0:
                # save map
                print(i)
                print('loss: ',loss)


            optimizer.zero_grad();
            loss.backward();
            optimizer.step();


#----------------------------------------------------------------------------------------------------------
    # save map
    np.save(savedir +'delta_recovery.npy', model_fit.param.reshape(1,256,256).cpu().detach().numpy()-5);


directory = '/home/yting/'

# lensing map
# my_file = directory + 'image_initial.npy'
# image = np.load(my_file)[0,:256,:256]

# Delta function
image = np.random.rand(256,256)<0.01

# synthesise
image_synthesis(image,
                learnable_param_list = [(100*2, 1e-0),(100*50, 1e-1)],
                savedir = directory)
