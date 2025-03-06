from pinf.datasets.log_likelihoods import log_p_2D_GMM

##################################################################################################
# 2D GMM power-scaling
##################################################################################################

def dS_dbeta_2D_GMM(x,gmm,device,beta = None):
    return - log_p_2D_GMM(x = x,beta_tensor = 1.0,device = device,gmm = gmm)

dS_dparam_dict = {
    "2D_GMM":dS_dbeta_2D_GMM,
}

dS_dparam_dict_multiple_parameters = {
    
}