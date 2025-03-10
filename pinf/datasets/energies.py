import torch
from typing import List,Union

from pinf.models.GMM import GMM
from pinf.datasets.log_likelihoods import (
    log_p_2D_ToyExample_two_parameters,
    log_p_2D_GMM,
    log_p_ScalarTheory
    )

##################################################################################################
# 2D GMM power-scaling
##################################################################################################

def S_2D_GMM(x:torch.tensor,beta:Union[float,torch.tensor],gmm:GMM,device:str)->torch.tensor:
    return - log_p_2D_GMM(x = x,beta_tensor = beta,device = device,gmm = gmm)

##################################################################################################
# 2D GMM with two external parameters
##################################################################################################

def S_2D_ToyExample_two_parameters(x:torch.Tensor,parameter_list:List[torch.Tensor|float],device:str)->torch.tensor:
    return - log_p_2D_ToyExample_two_parameters(x = x, parameter_list = parameter_list, device = device, Z = None)

##################################################################################################
# Scalar Theory
##################################################################################################

def S_ScalarTheory(mus,kappas,lambdas):

    '''
    True action function for 2D lattice.
    
    parameters:
        mus:        Initial states
        kappas:     Tensor containing the hopping-parameters
        lambdas:    Tensor containing the quadric-couplings

    returns:
        actions:    Containing the action of the different states
    '''

    actions = - log_p_ScalarTheory(
        x = mus,
        beta_tensor=kappas,
        lambdas=lambdas
    )

    return actions

S_dict = {
    "2D_GMM":S_2D_GMM,
    "ScalarTheory":S_ScalarTheory,
}

S_dict_multiple_parameters = {
    "2D_ToyExample_two_external_parameters":S_2D_ToyExample_two_parameters
}