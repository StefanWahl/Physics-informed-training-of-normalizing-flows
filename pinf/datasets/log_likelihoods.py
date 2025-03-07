import torch
from typing import List
from FrEIA.utils import force_to
import torch.distributions as D
import numpy as np
from pinf.models.GMM import GMM

##################################################################################################
# 2D GMM power-scaling
##################################################################################################

def log_p_2D_GMM(x:torch.tensor,beta_tensor:float,device:str,gmm:GMM)->torch.tensor:

    if isinstance(beta_tensor,float):
        beta_tensor = torch.ones(len(x),1).to(device) * beta_tensor

    log_prob = gmm.log_prob(x)*beta_tensor.squeeze()

    assert (log_prob.shape == torch.Size([len(x)]))

    return log_prob

##################################################################################################
# 2D GMM with two external parameters
##################################################################################################

def log_p_2D_ToyExample_two_parameters(x:torch.Tensor,parameter_list:List[torch.Tensor|float],device:str,Z:float = None)->torch.tensor:
    """
    Mixture of two gaussian modes with a variable relative weighting and power scaling.

    parameters:
        x:              Batch of evaluated points in data space
        parameter_list: Batches of condition values for each condition
        device:         Device on which the experiment runs
        Z:              Partition function

    returns:
        log_p:          Log-likelihood of the evaluated points
    """

    assert(len(parameter_list) == 2)

    if (Z is not None) and (not isinstance(parameter_list[0],float) or not isinstance(parameter_list[1],float)):
        raise ValueError("For normalized density evaluation only one set of parameters for each point is allowed")

    if isinstance(parameter_list[0],float):
        parameter_list[0] = parameter_list[0] * torch.ones(len(x)).to(device)

    if isinstance(parameter_list[1],float):
        parameter_list[1] = parameter_list[1] * torch.ones(len(x)).to(device)

    parameter_list_ =[ parameter_list[0].squeeze(),parameter_list[1].squeeze()]

    S_1 = torch.tensor([[1.0,-0.5],[-0.5,7.0]]).to(device)
    S_2 = torch.tensor([[1.0,0.5],[0.5,7.0]]).to(device)

    m_1 = torch.tensor([-4.0,0.0]).to(device)
    m_2 = torch.tensor([4.0,0.0]).to(device)

    p_1 = force_to(D.MultivariateNormal(loc = m_1,covariance_matrix=S_1),device)
    p_2 = force_to(D.MultivariateNormal(loc = m_2,covariance_matrix=S_2),device)

    a_1 = (p_1.log_prob(x) + parameter_list_[0].log()).reshape(-1,1)
    a_2 = (p_2.log_prob(x) + (1.0 - parameter_list_[0]).log()).reshape(-1,1)

    a = torch.cat((a_1,a_2),1)

    log_p = parameter_list_[1] * torch.logsumexp(a,1)

    assert(log_p.shape == torch.Size((len(x),)))

    if Z is not None:
        log_p = log_p - np.log(Z)

    return log_p


log_p_target_dict = {
    "2D_GMM":log_p_2D_GMM,
}

log_p_target_dict_multiple_parameters = {
    "2D_ToyExample_two_external_parameters":log_p_2D_ToyExample_two_parameters
}


