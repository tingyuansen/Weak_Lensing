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

    if mode == 'func':
        random_phase = np.random.normal(0,1,(M//2-1,N-1)) + np.random.normal(0,1,(M//2-1,N-1))*1j
        random_phase_left = (np.random.normal(0,1,(M//2-1)) + np.random.normal(0,1,(M//2-1))*1j)[:,None]
        random_phase_top = (np.random.normal(0,1,(N//2-1)) + np.random.normal(0,1,(N//2-1))*1j)[None,:]
        random_phase_middle = (np.random.normal(0,1,(N//2-1)) + np.random.normal(0,1,(N//2-1))*1j)[None,:]
        random_phase_corners = np.random.normal(0,1,3)
    if mode == 'image':
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
        gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_modulus*np.exp(1j*2*np.pi*gaussian_phase)))
    if mode == 'func':
        X = np.arange(0,M)
        Y = np.arange(0,N)
        Xgrid, Ygrid = np.meshgrid(X,Y)
        R = ((Xgrid-M/2)**2+(Ygrid-N/2)**2)**0.5
        gaussian_modulus = target(R)
        gaussian_modulus[M//2, N//2] = 0
        gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_modulus*gaussian_phase))

    data = np.fft.fftshift(np.real(gaussian_field))
    return data


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

            # initialize with GRF of same PS as target image
            if True:
                self.param = torch.nn.Parameter(
                    torch.from_numpy(
                        get_random_data(image, num_pixel,num_pixel).reshape(1,-1)
                    ).type(torch.cuda.FloatTensor) + 5
                )

            # initialize with white noise GRF
            if True:
                self.param = torch.nn.Parameter(torch.randn(1,num_pixel,num_pixel).type(torch.cuda.FloatTensor)*0.2+6)

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
            loss_L2 = ((((model_fit.param.reshape(1,num_pixel,num_pixel) - 5)**2).mean()**0.5 - \
                       ((image_GPU-5)**2).mean()**0.5) / ((image_GPU-5)**2).mean()**0.5 )**2

            # loss: L1
            loss_L1 = (( (model_fit.param.reshape(1,num_pixel,num_pixel) - 5).abs().mean() - (image_GPU-5).abs().mean() )\
                        /(image_GPU-5).abs().mean() )**2

            # loss: mean
            loss_mean = ((model_fit.param.reshape(1,num_pixel,num_pixel) - 5).mean() - (image_GPU-5).mean())**2

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
    np.save(savedir +'delta_recovery.npy', model_fit.param.reshape(1,num_pixel,num_pixel).cpu().detach().numpy()-5);


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
