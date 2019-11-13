# import packages
import torch
from torch.nn import Linear, LeakyReLU, MSELoss, Sequential
from torch.optim import Adam

import numpy as np
import sys

# parse input
ng = int(sys.argv[1])
training_size = int(sys.argv[2])

#=========================================================================================================
# import training set
Sx_tr = np.log10(np.load("Sx_training_expected_" + str(ng) + ".npy"))[:,1:]
Sx_tr = torch.from_numpy(Sx_tr).type(torch.cuda.FloatTensor)
y_tr = np.load('../weak_lensing_data/sparse_grid_final_512pix_y_train.npy')
y_tr = torch.from_numpy(y_tr).type(torch.cuda.FloatTensor)

# randomize
ind_shuffle = np.arange(y_tr.shape[0])
np.random.shuffle(ind_shuffle)
Sx_tr = Sx_tr[ind_shuffle,:][:training_size,:]
y_tr = y_tr[ind_shuffle,:][:training_size,:]

# standardize
mu_tr = Sx_tr.mean(dim=0)
std_tr = Sx_tr.std(dim=0)
Sx_tr = (Sx_tr - mu_tr) / std_tr

mu_y = y_tr.mean(dim=0)
std_y = y_tr.std(dim=0)
y_tr = (y_tr - mu_y) / std_y

#----------------------------------------------------------------------------------------------
# import validation set
Sx_te = np.log10(np.load("Sx_testing_expected_" + str(ng) + ".npy"))[:,1:]
y_te = np.load('../weak_lensing_data/sparse_grid_final_512pix_y_test.npy')

# randomize
ind_shuffle = np.arange(y_te.shape[0])
np.random.shuffle(ind_shuffle)
Sx_valid = Sx_te[ind_shuffle,:][:6000,:]
y_valid = y_te[ind_shuffle,:][:6000,:]

Sx_te = Sx_te[ind_shuffle,:][6000:,:]
y_te = y_te[ind_shuffle,:][6000:,:]

# convert into torch variables
Sx_valid = torch.from_numpy(Sx_valid).type(torch.cuda.FloatTensor)
y_valid = torch.from_numpy(y_valid).type(torch.cuda.FloatTensor)
Sx_te = torch.from_numpy(Sx_te).type(torch.cuda.FloatTensor)
y_te = torch.from_numpy(y_te).type(torch.cuda.FloatTensor)

# standardize
Sx_valid = (Sx_valid- mu_tr) / std_tr
y_valid = (y_valid - mu_y) / std_y
Sx_te = (Sx_te - mu_tr) / std_tr
y_te = (y_te - mu_y) / std_y



#=========================================================================================================
# input/out dimensions
num_input = Sx_tr.shape[-1]
num_output = y_tr.shape[-1]
num_neuron = 30

# define networks
model = Sequential(\
            Linear(num_input, num_neuron), LeakyReLU(),\
            Linear(num_neuron, num_neuron), LeakyReLU(),\
            Linear(num_neuron, num_output))
optimizer = Adam(model.parameters())
criterion = MSELoss()

#----------------------------------------------------------------------------------------------
# use cuda
model = model.cuda()
criterion = criterion.cuda()

#----------------------------------------------------------------------------------------------
# Number of signals to use in each gradient descent step (batch).
batch_size = training_size//10

# Number of epochs.
num_epochs = 1e4

# Learning rate for Adam.
lr = 1e-4

#----------------------------------------------------------------------------------------------
# break into batches
nsamples = Sx_tr.shape[0]
nbatches = nsamples // batch_size

# initiate counter
current_loss = np.inf
training_loss =[]
validation_loss = []

#----------------------------------------------------------------------------------------------
# train the network
for e in range(int(num_epochs)):

    # Randomly permute the data. If necessary, transfer the permutation to the
    # GPU.
    perm = torch.randperm(nsamples)
    perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take
    # one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (i+1) * batch_size]
        model.zero_grad()
        resp = model.forward(Sx_tr[idx])
        loss = criterion(resp, y_tr[idx])
        loss.backward()
        optimizer.step()

#----------------------------------------------------------------------------------------------
    # check periodically
    if e % 100 == 0:

        # calculate the response at the end of this epoch and the average loss.
        resp = model.forward(Sx_tr)
        avg_loss = criterion(resp, y_tr)

        resp_valid = model.forward(Sx_valid)
        avg_loss_valid = criterion(resp_valid, y_valid)

        resp_test = model.forward(Sx_te)
        avg_loss_test = criterion(resp_test, y_te)

        loss_data = avg_loss.detach().data.item()
        loss_valid_data = avg_loss_valid.detach().data.item()
        training_loss.append(loss_data)
        validation_loss.append(loss_valid_data)

        print('Epoch {}, training loss = {:1.3f}'.format(e, avg_loss*1e4),\
            'Epoch {}, validation loss = {:1.3f}'.format(e, avg_loss_valid*1e4))

        # record the weights and biases if the validation loss improves
        if loss_valid_data < current_loss:
            current_loss = loss_valid_data
            resp_best = resp_test*std_y + mu_y


#=========================================================================================================
# restore back to the original unit
y_te = y_te*std_y + mu_y

# save results
np.savez("results" + str(ng) + "_training_size=" + str(training_size) + ".npz",\
         resp_best=resp_best.cpu().detach().numpy(),\
         y_te=y_te.cpu().detach().numpy(),\
         training_loss = training_loss,\
         validation_loss = validation_loss)
