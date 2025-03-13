import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate

class HistogramDist():

    def __init__(self,data,n_bins,device = "cpu",range = None):

        self.device = device
        self.n_bins = n_bins

        #Get the densities and the borders of the bins
        densities,bin_edges = torch.histogram(input = data,bins= n_bins,range = range,density = True)

        #Add two additional bins with probability zero at the end of the density tensor to handle damples left and right of the histogram
        #[p_0,...p_N-1,0,0]
        #The last entry (index -1) represents the case of samples left of the histogram, the one before (index -2) represents the case of samples right of the histogram

        self.densities = torch.zeros(self.n_bins+2).to(self.device)
        self.densities[:self.n_bins] = densities

        #Get the parameters of the histogram
        self.h = bin_edges[1] - bin_edges[0]
        self.x_0 = bin_edges[0]

        #Get the left edges of the bins anad add the auxiliary bins for the border cases
        self.lefts = torch.zeros(self.n_bins+2).to(self.device)
        self.lefts[:self.n_bins+1] = bin_edges.to(self.device)
        self.lefts[-1] = self.x_0 - self.h

        #Get the mass left of each bin
        s = torch.cumsum(self.densities[:-2] * self.h,dim = -1) - self.densities[:-2] * self.h

        #As for the densities, add two auxiliary bins for the edge cases
        self.s = torch.zeros(self.n_bins+2).to(self.device)
        self.s[:self.n_bins] = s
        self.s[-1] = 0.0
        self.s[-2] = 1.0

    def sample(self,N_samples,eps = 1e-5):
        """
        Return samples following the histogram distribution by applying inversion sampling.

        parameters:
            N_samples: Number of samples to generate

        return: 
            x: Tensor of shape [N_sample] of samples
        """

        #Get teh uniformliy distributed samples
        u = torch.rand(N_samples).to(self.device)
        u = torch.clamp(u,0,1-eps)

        #Get the index of the bin in which the sample has to be sorted in
        idx = (torch.searchsorted(self.s,u,right = True) - 1.0).int()

        s_i = self.s[idx]
        p_i = self.densities[idx]
        lefts_i = self.lefts[idx]

        x = (u - s_i) / p_i + lefts_i

        #Avoid returning inf due samples at the right end of the histogram
        x = np.where(x < np.inf,x,self.lefts[-2] * torch.ones_like(x).to(self.device))

        return x

    def get_bin_index(self,x):
        """
        Get the index of the bin in which the evaluation points fall

        parameters:
            x: Tensor of shape [N] of evaluation points

        returns:
            idx: Tensor of shape [N] containing the bin indices
        """

        #print(f"x2: {x}")

        idx = torch.floor((x - self.x_0) / self.h)

        #print(f"I_1:{idx}")

        #Handle the border cases
        idx = torch.clamp(idx,min = -1,max = self.n_bins)
        #print(f"I_2:{idx}")

        idx = torch.where(idx == self.n_bins,-2 * torch.ones_like(idx).to(self.device),idx)
        #print(f"I_3:{idx}")

        return idx.int()

    def CDF(self,x):
        """
        Compute CDF of teh histogram distribution for the given values.

        parameters:
            x: Tensor of shape [N] of evaluation points

        returns:
            cdf: Tensor of shape [N] containing the CDF for the evaluation points
        """

        #Get the bins
        idx = self.get_bin_index(x)

        #Get the mass left of the bins
        s_i = self.s[idx]
        p_i = self.densities[idx]
        lefts_i = self.lefts[idx]

        cdf = s_i + p_i * (x - lefts_i)

        return cdf
    
    def __call__(self,x):
        """
        Compute PDF of teh histogram distribution for the given values.

        parameters:
            x: Tensor of shape [N] of evaluation points

        returns:
            pdf: Tensor of shape [N] containing the PDF for the evaluation points
        """
       
        idx = self.get_bin_index(x)
        pdf = self.densities[idx]


        return pdf

    def moment(self,n):
        """
        Compute the n th uncentred moment of the distribution.

        parameters:
            n: Order of the moment
        
        return.
            m: n th moment of the distribution
        """

        #Get the borders for the integration of the different bins 
        border_left = self.lefts[:self.n_bins]
        border_right = self.lefts[1:self.n_bins+1]

        I_bins = self.densities[:self.n_bins] * (border_right.pow(n+1) - border_left.pow(n+1)) / (n + 1)

        return I_bins.sum().item()
