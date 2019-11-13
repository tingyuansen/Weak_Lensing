# import packages
from kymatio import Scattering2D
import kymatio

import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data as utils

import time

import numpy as np

from kymatio import HarmonicScattering3D



#=========================================================================================================
# setup scattering
# number of pixels
num_pixel = 64

# define scattering
J_choice = 6
L_choice = 5
max_order_choice = 2
scattering = HarmonicScattering3D(J=J_choice, shape=(64,64,64),\
                          L=L_choice, max_order=max_order_choice)
scattering.cuda()


#=========================================================================================================
# main body of the script
def generate_image(ind):

    # target ccoefficients
    image_initial = torch.from_numpy(image).type(torch.cuda.FloatTensor) + 5.
    scattering_target = HarmonicScattering3D(J=J_choice, shape=(64,64,64),\
                              L=L_choice, max_order=max_order_choice)
    scattering_target.cuda()
    target_coeff = scattering_target(image_initial).view(x_image.shape[0],-1).log();

    # restore pre-calculated coefficients
    target_coeff = torch.from_numpy(np.load("scatter_coeff_max_order=2.npy")[ind,:]).type(torch.cuda.FloatTensor)
    CDF_t = torch.from_numpy(np.load("cdf_array.npy")[6,:]).type(torch.cuda.FloatTensor)

#----------------------------------------------------------------------------------------------------------
    # restore previous results
    if ind != 0:
        #pre_result = np.load("max_order=1.npy")
        pre_result = np.load("../max_order=2_ind=" + str(ind-1) + ".npy");

#----------------------------------------------------------------------------------------------------------
    # define mock image
    class model_image(nn.Module):
        def __init__(self):
            super(model_image, self).__init__()

            # star with the same image but with random phase
            #self.param = torch.nn.Parameter(
            #    torch.from_numpy(
            #        get_random_data(image[0], num_pixel, num_pixel).reshape(1,-1)
            #    ).type(torch.cuda.FloatTensor) + 5.
            #)

            if ind == 0:
                self.param = torch.nn.Parameter(
                    torch.from_numpy(get_random_data(np.concatenate((
                           np.concatenate((image_tiling[0],image_tiling[1]),0),
                           np.concatenate((image_tiling[2],image_tiling[3]),0)),1), num_pixel, num_pixel).reshape(1,-1)
                           ).type(torch.cuda.FloatTensor) + 5.
                    )
            else:
                # use previous results
                self.param = torch.nn.Parameter(
                            torch.from_numpy(pre_result).type(torch.cuda.FloatTensor)
                            )

#---------------------------------------------------------------------------------------------------------
    # learn with different training rate
    model_fit = model_image()
    learnable_param_list = [[100*50, 1e-2], [100*0, 1e-3], [100*0, 1e-4]]

    # loop over training rate
    for learnable_group in range(len(learnable_param_list)):

        # define learn hyper parameter
        num_step = learnable_param_list[learnable_group][0]
        learning_rate = learnable_param_list[learnable_group][1]

        # define optimizer
        optimizer = optim.SGD(model_fit.parameters(), lr=learning_rate)

        # optimize
        for i in range(int(num_step)):
            scattering_coeff = scattering(model_fit.param.reshape(1,num_pixel,num_pixel))\
                                    .mean(dim=(2,3))[0,:].log();
            loss_1 = ((target_coeff[1:]-scattering_coeff[1:])**2).sum(); # ignore the zeroth order (normalization)
            loss_2 = ((torch.sort(model_fit.param).values[0,::4] - CDF_t)**2).sum()/5.
            print(loss_1/loss_2) # making sure the two losses are of the same order
            loss = loss_1 + loss_2

#---------------------------------------------------------------------------------------------------------
            if i%50== 0:
                # save map
                #np.save("../results_step=" + str(i) + ".npy", model_fit.param.cpu().detach().numpy());
                #np.save("../scatter_coeff_step=" + str(i) + ".npy", scattering_coeff.cpu().detach().numpy());
                print(i, loss)
                print((target_coeff[1:]-scattering_coeff[1:]).abs()/target_coeff[1:].abs())

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

        np.save("../max_order=2_ind=" + str(ind) + ".npy", model_fit.param.cpu().detach().numpy());
        np.save("../max_order=2_scatter_coeff_ind=" + str(ind) + ".npy", scattering_coeff.cpu().detach().numpy());

#---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #main()

    # loop over all index
    for i in range(12):
        generate_image(i)
