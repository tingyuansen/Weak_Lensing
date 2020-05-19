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
def get_power_spectrum(target, bins):
    '''
    get the power spectrum of a given image
    '''
    M, N = target.shape
    modulus = torch.fft(torch.cat((target.reshape(M,N,1), torch.zeros((M,N,1)).type(torch.cuda.FloatTensor) ), 2), 2)
    modulus = (modulus[:,:,0]**2 + modulus[:,:,1]**2)**0.5
    modulus = torch.cat(
        ( torch.cat(( modulus[M//2:, M//2:], modulus[M//2:, :M//2] ), 0),
          torch.cat(( modulus[:M//2, M//2:], modulus[:M//2, :M//2] ), 0)
        ),1)
    X = np.arange(0,M)
    Y = np.arange(0,N)
    Xgrid, Ygrid = np.meshgrid(X,Y)
    R = ((Xgrid-M/2)**2+(Ygrid-N/2)**2)**0.5
    R = torch.from_numpy(R).type(torch.cuda.FloatTensor)
    R_range = torch.logspace(0.0, np.log10(M/2), bins).type(torch.cuda.FloatTensor)
    R_range = torch.cat((torch.tensor([0]).type(torch.cuda.FloatTensor), R_range))
    power_spectrum = torch.zeros(len(R_range)-1).type(torch.cuda.FloatTensor)
    for i in range(len(R_range)-1):
        select = (R >= R_range[i]) * (R<R_range[i+1])
        power_spectrum[i] = modulus[select].mean()
    return power_spectrum, R_range

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
                    savedir = 'drive/My Drive/Colab Notebooks/ST Yuan-Sen/',):

    image = image[None, :, :]

    torch.manual_seed(986)
    np.random.seed(986)

    # number of pixels
    num_pixel = 256
    J = 3
    L = 4

    # define scattering
    # scattering = Scattering2D(J=J, shape=(num_pixel,num_pixel), L=L, max_order=2)
    # scattering.cuda()

    # Using cumulative distribution function to set PDF constraint
    target_CDF = torch.from_numpy(np.sort(image.flatten())).type(torch.cuda.FloatTensor) + 5

    image_GPU = torch.from_numpy(image).type(torch.cuda.FloatTensor) + 5
    target_PS, _ = get_power_spectrum(image_GPU[0], 20)
    target_PS = target_PS.log()
    # target_ST = scattering(image_GPU).mean(dim=(2,3))[0,:].log();

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

            # initialize with smoothed target image
            if False:
                self.param = torch.nn.Parameter(
                    torch.from_numpy(
                        gaussian_filter(
                            np.random.randn(1,num_pixel,num_pixel),
                            (0,0.5,0.5)
                            )
                        ).type(torch.cuda.FloatTensor)*0.05 + 5
                    )
            # initialized with particular external image
            if False:
                train_result = np.load(savedir+'trained_result.npy')
                self.param = torch.nn.Parameter(torch.from_numpy(train_result.reshape(1,-1)).type(torch.cuda.FloatTensor))

            # others
            if False:
                pass
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

            # loss: power spectrum
            # PS, _ = get_power_spectrum(model_fit.param.reshape(num_pixel,num_pixel), 20)
            # PS = PS.log()
            # loss_PS = ((target_PS[:] - PS[:])**2).sum()

            # loss: scattering_coeff
            # ST = scattering(model_fit.param.reshape(1,num_pixel,num_pixel)).mean(dim=(2,3))[0,:].log();
            # loss_ST = ((target_ST - ST)**2).sum();

            # loss: CDF
            # CDF = torch.sort(torch.flatten(model_fit.param.reshape(1,num_pixel,num_pixel))).values
            # loss_CDF = ((target_CDF - CDF)**2).sum()

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
            loss = loss_L1 + loss_L2*0 #+ loss_mean

            if i%100== 0:
                # save map
                np.save(savedir + 'synthesis_results_step=' + str(i) + '.npy',
                        model_fit.param.reshape(1,num_pixel,num_pixel).cpu().detach().numpy()-5);
                np.save(savedir +'synthesis_results_final.npy',
                        model_fit.param.reshape(1,num_pixel,num_pixel).cpu().detach().numpy()-5);
                print(i)
                print('loss: ',loss)
                # print('loss_ST: ',loss_ST)
                # print('loss_PS: ',loss_PS)
                # print('loss_CDF: ',loss_CDF)
                print('loss_L1: ',loss_L1)
                print('loss_L2: ',loss_L2)
                print('loss_mean: ',loss_mean)
                # print('loss_bound: ',loss_bound)

                # print(target_PS, PS)
                # print(target_ST, ST)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();


#----------------------------------------------------------------------------------------------------------
    # save map
    np.save(savedir +'synthesis_results_final.npy', model_fit.param.reshape(1,num_pixel,num_pixel).cpu().detach().numpy()-5);


directory = '/home/yting'

# lensing map
# my_file = directory + 'image_initial.npy'
# image = np.load(my_file)[0,:256,:256]

# Delta function
image = np.random.rand(256,256)<0.01

# synthesise
image_synthesis(image,
                learnable_param_list = [(100*2, 1e-0),(100*50, 1e-1)],
                savedir = directory)
