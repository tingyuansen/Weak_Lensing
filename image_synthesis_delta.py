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

# define scattering
J_choice = 5
L_choice = 4
max_order_choice = 2
scattering = Scattering2D(J=J_choice, shape=(num_pixel,num_pixel),\
                          L=L_choice, max_order=max_order_choice)
scattering.cuda()


#=========================================================================================================
# main body of the script
def generate_image():

    # load an initial guess
    image = np.load("poisson_process_image.npy")[0:1,:,:]
    #CDF_t = torch.from_numpy(np.sort(image.flatten())).type(torch.cuda.FloatTensor) + 5.

#----------------------------------------------------------------------------------------------------------
    # target ccoefficients
    image_GPU = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    #print(image_initial.shape)
    # scattering_target = Scattering2D(J=J_choice, shape=(256,256),\
    #                               L=L_choice, max_order=max_order_choice)
    # scattering_target.cuda()
    # target_coeff = scattering_target(image_initial).mean(dim=(2,3))[0,:].log();

    # L2 norm
    #target_coeff = ((image_initial).abs()).mean()
    #print(target_coeff)

#----------------------------------------------------------------------------------------------------------
    # define mock image
    class model_image(nn.Module):
        def __init__(self):
            super(model_image, self).__init__()

            # star with the same image but with random phase
            self.param = torch.nn.Parameter(torch.rand(1,256,256).type(torch.cuda.FloatTensor))

            # self.param = torch.nn.Parameter(
            #     torch.from_numpy(
            #         np.random.uniform(size=(1,256,256))
            #     ).type(torch.cuda.FloatTensor)
            # )

#---------------------------------------------------------------------------------------------------------
    # learn with different training rate
    model_fit = model_image()
    learnable_param_list = [[100*50, 1e-3], [100*0, 1e-3], [100*0, 1e-4]]

    # loop over training rate
    for learnable_group in range(len(learnable_param_list)):

        # define learn hyper parameter
        # num_step = learnable_param_list[learnable_group][0]
        # learning_rate = learnable_param_list[learnable_group][1]
        #
        # # define optimizer
        # optimizer = optim.SGD(model_fit.parameters(), lr=learning_rate)

        num_step = learnable_param_list[learnable_group][0]
        learning_rate = learnable_param_list[learnable_group][1]

        # optimizer = optim.Adam(model_fit.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model_fit.parameters(), lr=learning_rate)

        # optimize
        for i in range(int(num_step)):
            #scattering_coeff = scattering(model_fit.param.reshape(1,num_pixel,num_pixel))\
            #                        .mean(dim=(2,3))[0,:].log();
            #loss = ((target_coeff[1:]-scattering_coeff[1:])**2).sum(); # ignore the zeroth order (normalization)
            #loss_2 = ((torch.sort(model_fit.param).values[0,::4] - CDF_t)**2).sum()/5.
            #print(loss_1/loss_2) # making sure the two losses are of the same order

            # loss: L2
            # loss = ((((model_fit.param.reshape(1,num_pixel,num_pixel))**2).mean()**0.5 - \
            #            ((image_initial)**2).mean()**0.5) / ((image_initial)**2).mean()**0.5 )**2

            # loss: L1
            #loss_L1 = (( (model_fit.param.reshape(1,num_pixel,num_pixel) - 5).abs().mean() - (image_GPU-5).abs().mean() )\
            #            /(image_GPU-5).abs().mean() )**2


            # scattering_coeff = ((model_fit.param).abs()).mean();
            # print(scattering_coeff)
            # loss = ((target_coeff-scattering_coeff).abs()).sum()

            # loss: L2
            loss_L2 = ((((model_fit.param.reshape(1,num_pixel,num_pixel) - 5)**2).mean()**0.5 - \
                       ((image_GPU-5)**2).mean()**0.5) / ((image_GPU-5)**2).mean()**0.5 )**2

            # loss: L1
            loss_L1 = (( (model_fit.param.reshape(1,num_pixel,num_pixel) - 5).abs().mean() - (image_GPU-5).abs().mean() )\
                        /(image_GPU-5).abs().mean() )**2

            # loss: mean
            loss_mean = ((model_fit.param.reshape(1,num_pixel,num_pixel) - 5).mean() - (image_GPU-5).mean())**2

            # loss_bound = (5 - model_fit.param.reshape(1,num_pixel,num_pixel)).mean() + \
            #              ( model_fit.param.reshape(1,num_pixel,num_pixel) - 10).mean()

            # loss: 1st moment
            # loss = loss_PS * 0 + loss_CDF * 0 + loss_bound * 0 +\
            #         loss_L1 + loss_L2 + loss_mean
            loss = loss_L1 + loss_L2 + loss_mean

#---------------------------------------------------------------------------------------------------------
            if i%50== 0:
                print(i, loss)
                #print((target_coeff[1:]-scattering_coeff[1:]).abs()/target_coeff[1:].abs())

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

    np.save("../delta_recovery.npy", model_fit.param.cpu().detach().numpy());

#---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    generate_image()
