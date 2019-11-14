# restore training set
import numpy as np

temp = np.load('../Zeldovich_Approximation.npz')
sim_z0 = temp["sim_z0"]
sim_z50 = temp["sim_z50"]

#-------------------------------------------------------------------------------------
# import packages
from kymatio import HarmonicScattering3D

# make scattering coefficients
J_choice = 6
L_choice = 5
max_order_choice = 2
scattering = HarmonicScattering3D(J=J_choice, shape=(64,64,64),\
                          L=L_choice, max_order=max_order_choice)
scattering.cuda()

import torch
x_image = torch.from_numpy(sim_z0).type(torch.cuda.FloatTensor)
scatter_coeff = scattering(x_image).view(x_image.shape[0],-1).cpu().detach().numpy()
print(scatter_coeff.shape)

# save results
np.save("scatter_coeff_3D_max_order=" + str(max_order_choice) + ".npy", scatter_coeff)
