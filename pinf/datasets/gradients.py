import torch
from FrEIA.utils import force_to
import torch.distributions as D
from typing import List,Union

from pinf.models.GMM import GMM
from pinf.datasets.log_likelihoods import log_p_2D_GMM
from pinf.datasets.energies import S_2D_ToyExample_two_parameters

##################################################################################################
# 2D GMM power-scaling
##################################################################################################

def dS_dbeta_2D_GMM(x:torch.tensor,gmm:GMM,device:str,beta:Union[float,torch.tensor] = None):
    return - log_p_2D_GMM(x = x,beta_tensor = 1.0,device = device,gmm = gmm)

##################################################################################################
# 2D GMM with two external parameters
##################################################################################################

def dS_2D_ToyExample_two_parameters_dalpha(x:torch.Tensor,parameter_list:List[torch.Tensor|float],device:str)->torch.tensor:
    
    assert(len(parameter_list) == 2)

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

    log_p_1_x = p_1.log_prob(x)
    log_p_2_x = p_2.log_prob(x)

    a_1 = (log_p_1_x + parameter_list_[0].log()).reshape(-1,1)
    a_2 = (log_p_2_x + (1.0 - parameter_list_[0]).log()).reshape(-1,1)

    a = torch.cat((a_1,a_2),1)

    r = - parameter_list_[1] * (log_p_1_x.exp() - log_p_2_x.exp()) / torch.logsumexp(a,1).exp()

    return r

def dS_2D_ToyExample_two_parameters_dbeta(x:torch.Tensor,parameter_list:List[torch.Tensor|float],device:str)->torch.tensor:

    parameter_list_new = [parameter_list[0],1.0]

    return S_2D_ToyExample_two_parameters(x = x,parameter_list = parameter_list_new,device = device)

##################################################################################################
# Scalar Theory
##################################################################################################

def dS_dkappa_ScalarTheory(mus,parameter_list:List[torch.Tensor|float] = None,device:str = None):

    actions = - 2  * torch.roll(input=mus,shifts=1,dims=2) * mus
    actions += - 2 * torch.roll(input=mus,shifts=1,dims=3) * mus

    actions = torch.sum(input=actions,dim = [1,2,3])

    return actions

dS_dparam_dict = {
    "2D_GMM":dS_dbeta_2D_GMM,
    "ScalarTheory":dS_dkappa_ScalarTheory,
}

dS_dparam_dict_multiple_parameters = {
    "dS_2D_ToyExample_two_parameters_dalpha":dS_2D_ToyExample_two_parameters_dalpha,
    "dS_2D_ToyExample_two_parameters_dbeta":dS_2D_ToyExample_two_parameters_dbeta
}