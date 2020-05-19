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
            self.param = torch.nn.Parameter(torch.rand(1,256,256).type(torch.cuda.FloatTensor))

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
            # scattering_coeff = ((model_fit.param).abs()).mean();
            # print(scattering_coeff)
            # loss = ((target_coeff-scattering_coeff).abs()).sum()

            # loss: L2
            # loss_L2 = ((((model_fit.param.reshape(1,num_pixel,num_pixel) - 5)**2).mean()**0.5 - \
            #            ((image_GPU-5)**2).mean()**0.5) / ((image_GPU-5)**2).mean()**0.5 )**2
            #
            # # loss: L1
            # loss_L1 = (( (model_fit.param.reshape(1,num_pixel,num_pixel) - 5).abs().mean() - (image_GPU-5).abs().mean() )\
            #             /(image_GPU-5).abs().mean() )**2
            #
            # # loss: mean
            # loss_mean = ((model_fit.param.reshape(1,num_pixel,num_pixel) - 5).mean() - (image_GPU-5).mean())**2
            # loss = loss_L1 + loss_L2 + loss_mean

            loss = (( (model_fit.param).abs().mean() - (image_GPU).abs().mean() )\
                        /(image_GPU).abs().mean())**2

#---------------------------------------------------------------------------------------------------------
            if i%50== 0:
                print(i, loss)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

    np.save("../delta_recovery.npy", model_fit.param.cpu().detach().numpy());

#---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    generate_image()
