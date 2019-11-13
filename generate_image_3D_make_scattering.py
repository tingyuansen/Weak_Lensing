# restore training set
import numpy as np

temp = np.load('../Zeldovich_Approximation.npz')
sim_z0 = temp["sim_z0"]
sim_z50 = temp["sim_z50"]

#-------------------------------------------------------------------------------------
# import packages
from kymatio import HarmonicScattering3D

# make scattering coefficients
J_choice = 5
L_choice = 4
max_order_choice = 1
scattering = HarmonicScattering3D(J=J_choice, shape=(64,64,64),\
                          L=L_choice, max_order=max_order_choice)
#scattering.cuda()

import torch
x_image = torch.from_numpy(sim_z0).type(torch.cuda.FloatTensor)
scatter_coeff = scattering(x_image).mean(dim=(2,3,4)).cpu().detach().numpy()

# save results
np.save("../scatter_coeff_max_order=" + str(max_order_choice) + ".npy", scatter_coeff)
