#!/usr/bin/env python
# coding: utf-8
# *Author: Dezso Ribli*

"""
Class and script for evaluation baseline weak lensing
descriptors: the power spectrum and peak counts.
"""


import numpy as np
# FFT segfault, fixed with this
# https://github.com/IntelPython/mkl_fft/issues/11
np.fft.restore_all()

from lenstools import ConvergenceMap
from astropy import units as u
import argparse 
import pickle
from utils import add_shape_noise, smooth


class WLDescriptor():
    """Weak lensing descriptor class"""
    
    def __init__(self, descriptor, nbins=20, ng=None, seed = 42, 
                 map_size=512, scale=3.5*60, smoothing=1.0, pinv=False,
                 bins=None, fiducial_params=(0.309, 0.816)):
        """ Initialize WL descriptor class with parameters."""
        self.descriptor = descriptor
        assert self.descriptor in ['power_spectrum','peak_counts']
        self.nbins = nbins  # bin edges
        self.bins = bins  # set bins
        self.ng = ng  # shape noise
        self.map_size = map_size  # map size in pixela
        self.A_pix = (float(scale)/map_size)**2  # pixel area in arcmins
        self.scale = float(scale)  # full map size in arcmins
        self.smoothing = smoothing  # Gaussian smoothing scale
        self.pseudoinverse = pinv  # pseudoinverse or not
        self.fiducial_params = fiducial_params  # fid params
        self.seed = seed  # fix random seed for reproducibility
        self.rng = np.random.RandomState(seed=seed)
        # set histogram edges
        
        
        
        
    def fit(self, x, omega_m_list, sigma_8_list):
        """Calculate the mean peak counts and covariances."""
        # first set bins if unset
        if self.bins is None:
            self.set_default_bins(x, omega_m_list, sigma_8_list)
        
        # Evaluate each map
        self.descriptor_list = {}  # dictionary for observable list
        for xi,o,s in zip(x, omega_m_list, sigma_8_list):  # loop maps
            im = self.process_map(xi)  # add noise, apply smoothing
            d = self.descript(im)  # calculate descriptor
            # add to dictionary
            if (o,s) in self.descriptor_list:
                self.descriptor_list[(o,s)].append(d)
            else:
                self.descriptor_list[(o,s)] = [d]
                    
        # calculate means
        self.mean_descriptor = {}
        for k,v in self.descriptor_list.iteritems():
            self.mean_descriptor[k] = np.mean(v,axis=0)
            
        # Calculate inverse covariances
        # Empty bins, pseaudoinverse
        self.inv_cov = {}
        for k, v in self.descriptor_list.iteritems():
            # create matrix
            vm = np.vstack(v).T
            # correction
            c = (len(v) - len(self.bins)-1 -2)/float(len(v)-1)
            # calcuate the pseudoinverse of the covariance
            if self.pseudoinverse:
                self.inv_cov[k] = c * np.linalg.pinv(np.cov(vm))
            else:
                self.inv_cov[k] = c * np.linalg.inv(np.cov(vm))

                
    def process_map(self, x_in):
        """Process data."""
        x = np.array(x_in,copy=True)            
        if self.ng:  # add noise if ng is not None
            x = add_shape_noise(x, self.A_pix, self.ng, self.rng)
        if self.smoothing:  # smooth if smoothing is not None
            x = smooth(x, self.smoothing, self.scale)
        return x
    

    def descript(self, im):
        """Apply descriptor on convergence map."""
        if self.descriptor == 'peak_counts':
            return self.peak_count(im)
        elif self.descriptor == 'power_spectrum':
            return self.power_spectrum(im)


    def power_spectrum(self, im):
        """Calculate power spectrum."""
        conv_map = ConvergenceMap(im, angle=u.degree * 3.5)
        l,Pl = conv_map.powerSpectrum(self.bins)
        return Pl


    def peak_count(self, im):
        """Peak counting statistics"""
        peaks = self.find_peaks(im)  # find peaks  
        vals = im[1:-1,1:-1][peaks]  # get the values for peaks        
        hp = np.histogram(vals, bins=self.bins)[0]  # make histogram
        return hp


    def find_peaks(self, im):
        """Find peaks in bw image."""
        p =  im[1:-1,1:-1]>im[:-2,:-2]  # top left
        p &= im[1:-1,1:-1]>im[:-2,1:-1]  # top center  
        p &= im[1:-1,1:-1]>im[:-2,2:]  # top right
        p &= im[1:-1,1:-1]>im[1:-1,:-2]  # center left 
        p &= im[1:-1,1:-1]>im[1:-1,2:]  # center right 
        p &= im[1:-1,1:-1]>im[2:,:-2]  # bottom left
        p &= im[1:-1,1:-1]>im[2:,1:-1]  # bottom center
        p &= im[1:-1,1:-1]>im[2:,2:]   # bottom right
        return p

    
    def set_default_bins(self, x, omega_m_list, sigma_8_list):
        """Caluclate default bin edges."""
        if self.descriptor == 'power_spectrum':  # as in Gupta et al
            if self.map_size == 1024:
                self.bins = np.logspace(np.log10(100.), np.log10(75000.), 
                                        self.nbins+1)
            elif self.map_size == 512:  # cut large l for coarser grid
                self.bins = np.logspace(np.log10(100.), np.log10(37500.), 
                                        self.nbins+1)
        elif self.descriptor == 'peak_counts':
            # deinfe bins edges as in Gupta et al
            sn_edges         = np.linspace(-2.0, 12.0, self.nbins+1)
            kappa_rms = self.get_kappa_rms(x, omega_m_list, sigma_8_list)
            #kappa_edges   = 0.01767875 * sn_edges
            kappa_edges   = kappa_rms * sn_edges
            self.bins = kappa_edges
            

    def get_kappa_rms(self, x, omega_m_list, sigma_8_list):
        """Return the mean kappa  r.m.s. of fiducial maps."""
        kappa_rms = []
        for xi,o,s in zip(x, omega_m_list, sigma_8_list):  # loop maps
            if o == self.fiducial_params[0] and s==self.fiducial_params[1]:
                im = smooth(xi, self.smoothing, self.scale)  # apply smoothing
                kappa_rms.append(im.std())  # get kappa rms for a map
        return np.mean(kappa_rms)  # return mean kappa rms
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", type=str, default = 'power_spectrum',
                        choices=['power_spectrum','peak_counts'],
                        help="Desciptor")
    parser.add_argument("--map-size", type=int, choices=[512, 1024], default=512,
                        help="Map size in pixels")
    parser.add_argument("--shape-noise", type=float,
                        help="Shape noise, parametrized by the galaxy denisty, \
                        number of galaxies per square arcmin.")
    parser.add_argument("--smoothing", type=float, default = 1.0,
                        help="Smoothing with a Gaussian kernel, width should be \
                        defined in arcmins.")
    parser.add_argument("--grfized", 
                        help="Use equivalent Gaussian random field maps instead of \
                        physical maps",
                        action="store_true")
    parser.add_argument("--pinv", help="Use the pseudoinverse of the covariance.",
                        action="store_true")
    parser.add_argument("--n-bins", default=20, type=int,
                        help=" Number of bins in descriptor" )
    args = parser.parse_args()
    
    # create a run name
    RUN_NAME =  args.descriptor+ '_pix'+str(args.map_size)
    RUN_NAME+='_noise'+str(args.shape_noise) + '_Nbins'+str(args.n_bins)
    RUN_NAME+='_smoothing%.1f'%args.smoothing
    if args.grfized:
        RUN_NAME += '_grfized'
    
    # load data
    X_fn = '../../data/columbia_data_final_v2_fnsorted_'+str(args.map_size)+'pix.npy'
    y_fn = '../../data/columbia_data_final_v2_fnsorted_y.npy'
    if args.grfized:
        X_fn = '../../data/columbia_data_fnsorted_512pix_GRF_A.npy'
    X, y = np.load(X_fn), np.load(y_fn)

    # create weak lensing descriptor model
    wldesc = WLDescriptor(descriptor=args.descriptor, nbins=args.n_bins,
                          ng=args.shape_noise, map_size=args.map_size,
                          pinv = args.pinv, smoothing=args.smoothing)
    # fit it
    wldesc.fit(X, y[:,0], y[:,1])
    # save it
    with open('results/wldescriptor_'+RUN_NAME+'.p','wb') as fh:
        pickle.dump(wldesc,fh)
