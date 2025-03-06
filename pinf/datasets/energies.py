from pinf.datasets.log_likelihoods import log_p_2D_GMM

##################################################################################################
# 2D GMM power-scaling
##################################################################################################

def S_2D_GMM(x,beta,gmm,device):
    return - log_p_2D_GMM(x = x,beta_tensor = beta,device = device,gmm = gmm)

S_dict = {
    "2D_GMM":S_2D_GMM,
}

S_dict_multiple_parameters = {
    
}