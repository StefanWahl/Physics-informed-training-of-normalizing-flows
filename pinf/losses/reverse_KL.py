import torch
from typing import Callable,Dict,List
import numpy as np

#####################################################################
# Multiple external conditions
#####################################################################

class Objective_reverse_KL_Multiple_Parameters():
    def __init__(self,
                 parameter_limit_list:List[List[float]],
                 sample_param_in_log_space_list:List[bool],
                 log_p_target:Callable,
                 log_p_target_kwargs:Dict,
                 device:str,
                 bs:int,
                 )->None:

        """
        parameters:
            parameter_limit_list:           List of Lists containing the lower and upper limit for each dimension in the parameter space
            sample_param_in_log_space_list: List of bools determining how to sample the condition values 
            log_p_target                    Log likelihood of the (unnormalized) ground truth distribution
            log_p_target_kwargs:            Additional arguments for the ground truth log likelihood
            device:                         Name of the device on which the experiment runs
            bs:                             Batch size
        """
    
        self.parameter_limit_list = parameter_limit_list
        self.sample_param_in_log_space_list = sample_param_in_log_space_list
        self.log_p_target = log_p_target
        self.log_p_target_kwargs = log_p_target_kwargs
        self.bs = bs
        self.device = device
        self.iteration = 1

        print("*********************************************************************************************")
        print("Use class 'Objective_reverse_KL_Multiple_Parameters'")
        print("*********************************************************************************************")

    def __call__(self,INN,epoch,get_eval_points,logger = None)->torch.Tensor:
        """
        Compute the TS-PINF loss

        parameters:
            INN:                The INN model
            epoch:              The current epoch
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            logger:             The logger for the loss

        returns:
            rev_KL:               The temperature scaling loss
        """

        parameter_list = []

        # Get conditions
        for i in range(len(self.parameter_limit_list)):

            # Sample conditions uniformly in the log space
            if self.sample_param_in_log_space_list[i]:
                log_param_tensor_i = (np.log(self.parameter_limit_list[i][1]) - np.log(self.parameter_limit_list[i][0])) * torch.rand([self.bs,1]).to(self.device) + np.log(self.parameter_limit_list[i][0])
                param_tensor_i = log_param_tensor_i.exp()

            else:
                param_tensor_i = (self.parameter_limit_list[i][1] - self.parameter_limit_list[i][0]) * torch.rand([self.bs,1]).to(self.device) + self.parameter_limit_list[i][0]
            
            parameter_list.append(param_tensor_i.to(self.device))

        # Get INN samples
        x = INN.sample(self.bs,parameter_list = parameter_list)

        # Evaluate the samples on the ground truth log likelihood
        log_p_target_x = self.log_p_target(x = x,parameter_list = parameter_list,device = self.device,**self.log_p_target_kwargs)

        # Get the log-likelihood under the INN model
        log_p_theta_x = INN.log_prob(x,parameter_list = parameter_list)

        # Filter out invalid values
        mask_A = torch.isfinite(log_p_target_x)
        mask_B = torch.isfinite(log_p_theta_x)

        mask = mask_A * mask_B

        # Get the ratios
        r = log_p_theta_x[mask] - log_p_target_x[mask]
        rev_KL = r.mean()

        # Log the ratio of valid samples
        valid_r = mask.sum() / len(mask)

        logger.experiment.add_scalar(f"metadata/valid_r",valid_r,self.iteration)
        self.iteration += 1

        return rev_KL

#####################################################################
# One external condition
#####################################################################

class Objective_reverse_KL():
    def __init__(self,beta_min:float,
                 beta_max:float,
                 log_p_target:Callable,
                 log_p_target_kwargs:Dict,
                 device:str,
                 bs:int,
                 )->None:

        """
        parameters:
            beta_min:                       The minimal condition value
            beta_max:                       The maximal condition value
            log_p_target                    Log-likelihood of the (unnormalized) ground-truth distribution
            log_p_target_kwargs:            Additional arguments for the ground truth-log-likelihood
            device:                         Name of the device on which the experiment runs
            bs:                             Batch size
        """
    
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.log_p_target = log_p_target
        self.log_p_target_kwargs = log_p_target_kwargs
        self.bs = bs
        self.device = device
        self.iteration = 1

        print("*********************************************************************************************")
        print("Use class 'Objective_reverse_KL'")
        print("*********************************************************************************************")

    def __call__(self,INN,epoch,get_eval_points,logger = None)->torch.Tensor:
        """
        Compute the TS-PINF loss

        parameters:
            INN:                The INN model
            epoch:              The current epoch
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            logger:             The logger for the loss

        returns:
            rev_KL:               The temperature scaling loss
        """

        # Get condition value uniformly from the log space
        log_beta_tensor = (np.log(self.beta_max) - np.log(self.beta_min)) * torch.rand([self.bs,1]).to(self.device) + np.log(self.beta_min)
        beta_tensor = log_beta_tensor.exp()

        # Get INN samples
        x = INN.sample(self.bs,beta_tensor = beta_tensor)

        # Evaluate the samples on the ground-truth log-likelihood
        log_p_target_x = self.log_p_target(x = x,beta_tensor = beta_tensor,device = self.device,**self.log_p_target_kwargs)

        #Get the log-likelihood under the INN model
        log_p_theta_x = INN.log_prob(x,beta_tensor)

        # Filter out invalid values
        mask_A = torch.isfinite(log_p_target_x)
        mask_B = torch.isfinite(log_p_theta_x)

        mask = mask_A * mask_B

        # Get the ratios
        r = log_p_theta_x[mask] - log_p_target_x[mask]
        rev_KL = r.mean()

        # Log the ratio of valid samples
        valid_r = mask.sum() / len(mask)

        logger.experiment.add_scalar(f"metadata/valid_r",valid_r,self.iteration)
        self.iteration += 1

        return rev_KL
    