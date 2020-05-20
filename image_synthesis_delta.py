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
# setup scattering
# number of pixels
num_pixel = 256


#=========================================================================================================
# main body of the script
def generate_image():

    # load an initial guess
    image = np.load("poisson_process_image.npy")[None,:,:]

    # target ccoefficients
    image_GPU = torch.from_numpy(image).type(torch.cuda.FloatTensor)


#----------------------------------------------------------------------------------------------------------
    # define mock image
    class model_image(nn.Module):
        def __init__(self):
            super(model_image, self).__init__()
            self.param = torch.nn.Parameter(torch.rand(1,256,256)*100..type(torch.cuda.FloatTensor))

#---------------------------------------------------------------------------------------------------------
    # learn with different training rate
    model_fit = model_image()
    learnable_param_list = [(100*2, 1e-0),(100*500, 1e-1)]

    # loop over training rate
    for learnable_group in range(len(learnable_param_list)):
        num_step = learnable_param_list[learnable_group][0]
        learning_rate = learnable_param_list[learnable_group][1]

        # optimizer = optim.Adam(model_fit.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model_fit.parameters(), lr=learning_rate)

        # optimize
        for i in range(int(num_step)):
            # loss: L2

            loss_L2 = ((((1./(1.+(-1*model_fit.param).exp()))**2).mean()**0.5 - \
                       (image_GPU**2).mean()**0.5) / (image_GPU**2).mean()**0.5 )**2

            # loss: L1
            loss_L1 = ( ((1./(1.+(-1*model_fit.param).exp())).abs().mean() - image_GPU.abs().mean() )\
                        /image_GPU.abs().mean() )**2
            #
            # # loss: mean
            loss_mean = ((1./(1.+(-1*model_fit.param).exp())).mean() - image_GPU.mean())**2

            loss =  loss_L2 + loss_L1 + loss_mean


#---------------------------------------------------------------------------------------------------------
            if i%50== 0:
                print(i, loss)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

    np.save("../delta_recovery.npy", (1./(1.+(-1*model_fit.param).exp())).cpu().detach().numpy());

#---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    generate_image()
