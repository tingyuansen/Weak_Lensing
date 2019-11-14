# import packages
import kymatio
from kymatio import HarmonicScattering3D

import torch.nn as nn
import torch.optim as optim
import torch

import time

import numpy as np


#=========================================================================================================
# setup scattering
# number of pixels
num_pixel = 64

# define scattering
J_choice = 6
L_choice = 3
max_order_choice = 1
scattering = HarmonicScattering3D(J=J_choice, shape=(64,64,64),\
                          L=L_choice, max_order=max_order_choice)
scattering.cuda()


#=========================================================================================================
# restore scattering coefficient
target_coeff = np.load("scatter_coeff_3D_max_order=1.npy")[0,:]
target_coeff = torch.from_numpy(np.log(target_coeff)).type(torch.cuda.FloatTensor)

# restore data
temp = np.load('../Zeldovich_Approximation.npz')
sim_z0 = temp["sim_z0"]
sim_z50 = temp["sim_z50"]

CDF_t = torch.from_numpy(np.sort(sim_z0[0,:,:,:].flatten())).type(torch.cuda.FloatTensor)


#=========================================================================================================
# main body of the script
def main():

    # define mock image
    class model_image(nn.Module):
        def __init__(self):
            super(model_image, self).__init__()

            # start with a random image
            self.param = torch.nn.Parameter(
                            torch.from_numpy(sim_z50[0:1,:,:,:]).type(torch.cuda.FloatTensor)
                        )

            # random image
            # star with the same image but with random phase
            #self.param = torch.nn.Parameter(
            #    torch.from_numpy(
            #        get_random_data(image[0], num_pixel, num_pixel).reshape(1,-1)
            #    ).type(torch.cuda.FloatTensor) + 5.
            #)

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
            scattering_coeff = scattering(model_fit.param).view(-1).log();
            loss_1 = ((target_coeff[1:]-scattering_coeff[1:])**2).sum();
            loss_2 = ((torch.sort(model_fit.param.flatten()).values - CDF_t)**2).sum()
            print(loss_1/loss_2)
            loss = loss_1 + loss_2

#---------------------------------------------------------------------------------------------------------
            if i%50== 0:
                # save map
                np.save("../results_step=" + str(i) + ".npy", model_fit.param.cpu().detach().numpy());
                print(i, loss)
                print((target_coeff-scattering_coeff).abs()/target_coeff.abs())

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

#---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
