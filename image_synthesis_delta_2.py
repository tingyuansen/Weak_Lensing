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
    image_GPU = torch.from_numpy(image).type(torch.cuda.FloatTensor)

#----------------------------------------------------------------------------------------------------------
    # define mock image
    class model_image(nn.Module):
        def __init__(self):
            super(model_image, self).__init__()
            self.param = torch.nn.Parameter(torch.rand(1,256,256).type(torch.cuda.FloatTensor))

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
            loss = (( (model_fit.param.reshape(1,256,256)).abs().mean() - (image_GPU).abs().mean() )\
                        /(image_GPU).abs().mean())**2

            if i%100== 0:
                print(i)
                print('loss: ',loss)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();


#----------------------------------------------------------------------------------------------------------
    # save map
    np.save(savedir +'delta_recovery.npy', model_fit.param.reshape(1,256,256).cpu().detach().numpy());


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
