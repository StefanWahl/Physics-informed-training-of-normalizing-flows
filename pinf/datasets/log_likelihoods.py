from pinf.models.GMM import GMM
import torch

##################################################################################################
# 2D GMM power-scaling
##################################################################################################

def log_p_2D_GMM(x:torch.tensor,beta_tensor:float,device:str,gmm:GMM):

    if isinstance(beta_tensor,float):
        beta_tensor = torch.ones(len(x),1).to(device) * beta_tensor

    log_prob = gmm.log_prob(x)*beta_tensor.squeeze()

    assert (log_prob.shape == torch.Size([len(x)]))

    return log_prob

log_p_target_dict = {
    "2D_GMM":log_p_2D_GMM,
}

log_p_target_dict_multiple_parameters = {
    
}