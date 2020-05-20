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
def get_random_data(target, M, N, mode='image'):
    '''
    get a gaussian random field with the same power spectrum as the image 'target' (in the 'image' mode),
    or with an assigned power spectrum function 'target' (in the 'func' mode).
    '''
    random_phase = np.random.rand(M//2-1,N-1)
    random_phase_left = np.random.rand(M//2-1)[:,None]
    random_phase_top = np.random.rand(N//2-1)[None,:]
    random_phase_middle = np.random.rand(N//2-1)[None,:]
    random_phase_corners = np.random.randint(0,2,3)/2
    gaussian_phase = np.concatenate((
                      np.concatenate((random_phase_corners[1][None,None],
                                      random_phase_left,
                                      random_phase_corners[2][None,None],
                                      -random_phase_left[::-1,:],
                                    ),axis=0),
                      np.concatenate((np.concatenate((random_phase_top,
                                                      random_phase_corners[0][None,None],
                                                      -random_phase_top[:,::-1],
                                                    ),axis=1),
                                      random_phase,
                                      np.concatenate((random_phase_middle,
                                                      np.array(0)[None,None],
                                                      -random_phase_middle[:,::-1],
                                                    ),axis=1),
                                      -random_phase[::-1,::-1],
                                    ),axis=0),
                                    ),axis=1)


    if mode == 'image':
        gaussian_modulus = np.abs(np.fft.fftshift(np.fft.fft2(target)))
    if mode == 'func':
        X = np.arange(0,M)
        Y = np.arange(0,N)
        Xgrid, Ygrid = np.meshgrid(X,Y)
        gaussian_modulus = target(((Xgrid-M/2)**2+(Ygrid-N/2)**2)**0.5)

    gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_modulus*np.exp(1j*2*np.pi*gaussian_phase)))
    data = np.fft.fftshift(np.real(gaussian_field))
    return data


#=========================================================================================================
# setup scattering
# number of pixels
num_pixel = 512

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
    image = np.load("image_initial.npy")[0:1,:,:]
    CDF_t = torch.from_numpy(np.sort(image.flatten())).type(torch.cuda.FloatTensor) + 5.

#----------------------------------------------------------------------------------------------------------
    # target ccoefficients
    image_initial = torch.from_numpy(image).type(torch.cuda.FloatTensor) + 5.
    scattering_target = Scattering2D(J=J_choice, shape=(512,512),\
                                  L=L_choice, max_order=max_order_choice)
    scattering_target.cuda()
    target_coeff = scattering_target(image_initial).mean(dim=(2,3))[0,:].log();

#----------------------------------------------------------------------------------------------------------
    # define mock image
    class model_image(nn.Module):
        def __init__(self):
            super(model_image, self).__init__()

            # star with the same image but with random phase
            self.param = torch.nn.Parameter(
               torch.from_numpy(
                   get_random_data(image[0], num_pixel, num_pixel).reshape(1,-1)
               ).type(torch.cuda.FloatTensor) + 5.
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
            loss_2 = ((torch.sort(model_fit.param).values[0,:] - CDF_t)**2).sum()/5.
            print(loss_1/loss_2) # making sure the two losses are of the same order
            loss = loss_1 + loss_2

#---------------------------------------------------------------------------------------------------------
            if i%50== 0:
                print(i, loss)
                print((target_coeff[1:]-scattering_coeff[1:]).abs()/target_coeff[1:].abs())

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

        np.save("../max_order=2.npy", model_fit.param.cpu().detach().numpy());
        np.save("../max_order=2_scatter_coeff.npy", scattering_coeff.cpu().detach().numpy());

#---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    generate_image()
