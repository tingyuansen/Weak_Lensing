# restore training set
import numpy as np
y = np.load('sparse_grid_final_512pix_y_test.npy')

choose = [np.where((y[:,0] > 0.7) == 1)[0],
          np.where((np.abs(y[:,0]-0.6) < 0.05)*(np.abs(y[:,1]-0.4) < 0.05) == 1)[0],
          np.where((np.abs(y[:,0]-0.57) < 0.05)*(np.abs(y[:,1]-0.5) < 0.05))[0],
          np.where((np.abs(y[:,0]-0.5) < 0.05)*(np.abs(y[:,1]-0.6) < 0.05))[0],
          np.where((np.abs(y[:,0]-0.45) < 0.03)*(np.abs(y[:,1]-0.7) < 0.05))[0],
          np.where((np.abs(y[:,0]-0.4) < 0.03)*(np.abs(y[:,1]-0.75) < 0.05))[0],
          np.where((np.abs(y[:,0]-0.37) < 0.03)*(np.abs(y[:,1]-0.8) < 0.05))[0],
          np.where((np.abs(y[:,0]-0.3) < 0.01)*(np.abs(y[:,1]-0.8) < 0.02))[0],
          np.where((np.abs(y[:,0]-0.3) < 0.01)*(np.abs(y[:,1]-0.84) < 0.02))[0],
          np.where((np.abs(y[:,0]-0.25) < 0.02)*(np.abs(y[:,1]-0.9) < 0.03))[0],
          np.where((np.abs(y[:,0]-0.22) < 0.02)*(np.abs(y[:,1]-1.0) < 0.03))[0],
          np.where((y[:,0] < 0.21)*(y[:,1] > 1.05))[0]]


# import packages
from kymatio import Scattering2D

# make scattering coefficients
J_choice = 5
L_choice = 1
max_order_choice = 1
scattering = Scattering2D(J=J_choice, shape=(512,512),\
                          L=L_choice, max_order=max_order_choice)
scattering.cuda()

import torch
scatter_coeff = []
for i in range(len(choose)):
    print(i)
    x_image = np.load('../weak_lensing_data/sparse_grid_final_512pix_x_test.npy')[choose[i],:,:,0]
    x_image = torch.from_numpy(x_image).type(torch.cuda.FloatTensor) + 5.
    scatter_coeff.append(np.median(scattering(x_image).mean(dim=(2,3)).log().cpu().detach().numpy(), axis=0))
scatter_coeff = np.array(scatter_coeff)

# save results
np.save("scatter_coeff_max_order=1.npy", scatter_coeff)
