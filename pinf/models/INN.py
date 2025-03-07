import FrEIA.framework as Ff
import torch
from FrEIA.utils import force_to
import torch.distributions as D
import numpy as np
from typing import Union,Callable,List

def _beta_processing_ignore_beta(beta_tensor:torch.tensor)-> None:
    """
    Ignore the input tensor. Use for unconditional INN
    """
    return None

def _beta_processing_log(beta_tensor:torch.tensor)-> torch.tensor:
    """
    Return the logarithm of the input tensor
    """
    return beta_tensor.log()

_beta_processing_dict = {
    "log_beta":_beta_processing_log,
    "ignore_beta":_beta_processing_ignore_beta
}

##############################################################
# Wrapper for INN operations
##############################################################

class INN_Model():
    def __init__(self,d:int,inn:Ff.InvertibleModule,device:str,latent_mode:str = "standard_normal",process_beta_mode:str = "log_beta",embedding_model:Callable=None)->None:
        """
        This is a wrapper class to handle all the operations on the distribution defined by the INN.

        parameters:
            d:                  Dimensionality fo the data space
            inn:                Invertible function
            device:             Device to run the code on 
            latent_mode:        Mode of the latent distribution of the INN
            process_beta_mode:  How to preprocess the condition passed to the INN
            embedding_model:    Function mapping the beta_tensor to a d'dimensional embedding
        """

        self.inn = inn
        self.device = device

        if latent_mode == "standard_normal":    
            self.p_0 = force_to(D.MultivariateNormal(torch.zeros(d).to(device), torch.eye(d)),device)
        
        self.latent_mode = latent_mode
        self.d = d
        self.process_beta_mode = process_beta_mode

        # Use learneable processing of the condition
        if self.process_beta_mode == "learnable": 
            print("Learable condition embedding")
            self.beta_processing_function = embedding_model

        else:
            "Use standard condition embedding"
            self.beta_processing_function = _beta_processing_dict[self.process_beta_mode]

    def load_state_dict(self,path:str)->None:
        """
        Load stored parameters.

        parameters:
            path:   Location of the stored parameters.
        """

        state_dict = torch.load(path,weights_only = False)["state_dict"]

        print("Load state dict for invertible function")

        self.inn.load_state_dict(state_dict=state_dict["INN"])

        if self.process_beta_mode == "learnable": 
            print("Load state dict for embedding model")
            self.beta_processing_function.load_state_dict(state_dict=state_dict["Embedder"])

    def eval(self):
        """
        Set model to evaluation mode.
        """

        self.inn.eval()

        if self.process_beta_mode == "learnable":
            self.beta_processing_function.eval()

    def train(self,b:bool=True):
        """
        Set model to training mode.
        """

        self.inn.train(b)

        if self.process_beta_mode == "learnable":
            self.beta_processing_function.train(b)
    
    def log_prob_p_0(self,z_tensor:torch.tensor,beta_tensor:torch.tensor)->torch.tensor:
        """
        Compute the log-likelihood of the latent distribution.

        parameters:
            z_tensor:       Latent code
            beta_tensor:    Condition

        return:
            log_p_z:        Log-likelihood of the latent code 
        """

        if self.latent_mode == "standard_normal":
            log_p_z = self.p_0.log_prob(z_tensor)
        
        elif self.latent_mode == "temperature_scaling":
            log_p_z = self.d / 2 * torch.log(beta_tensor.reshape(-1) / (2 * np.pi)) - z_tensor.pow(2).sum(-1) * beta_tensor.reshape(-1) / 2

        return log_p_z

    def _beta_processing(self,beta_tensor:torch.tensor)->torch.tensor:
        """
        Compute the condition of the INN

        parameters:
            beta_tensor:            Plain condition

        return:
            beta_tensor_processed:  Processed condition

        """
        beta_tensor_processed = self.beta_processing_function(beta_tensor)

        if beta_tensor_processed is not None:
            return [beta_tensor_processed]
        else:
            return beta_tensor_processed

    def log_prob(self,x:torch.tensor,beta_tensor:Union[float,torch.tensor])->torch.tensor:
        """
        Compute the log-likelihood of data points.

        parameters:
            x:              Data points to evaluate
            beta_tensor:    Condition

        return:
            log_prob_x:     log-likelihood of x
        """

        # If only a float is given for the inverse temperature, use it for the whole batch
        if isinstance(beta_tensor,float):
            beta_tensor = torch.ones([x.shape[0],1]).to(self.device).to(self.device) * beta_tensor

        z,jac = self.inn(x,self._beta_processing(beta_tensor),rev=False)

        p_0_z = self.log_prob_p_0(z_tensor = z,beta_tensor = beta_tensor)
        
        log_prob_x =  p_0_z + jac

        return log_prob_x

    def sample(self,n_samples:int,beta_tensor:Union[float,torch.tensor])-> torch.tensor:
        """
        Generate Samples following the distribution defined by the INN.

        parameters:
            n_samples:      Number of samples to generate
            beta_tensor:    Condition

        return:
            x:              Samples following the distribution defined by the INN
        """
        
        # If only a float is given for the inverse temperature, use it for the whole batch
        if isinstance(beta_tensor,float):
            beta_tensor = torch.ones([n_samples,1]).to(self.device).to(self.device) * beta_tensor

        if self.latent_mode == "standard_normal":
            z = self.p_0.sample([n_samples])
        
        elif self.latent_mode == "temperature_scaling":
            z = torch.randn([n_samples,self.d]).to(self.device) * 1 / beta_tensor.sqrt()

        x,_ = self.inn(z,self._beta_processing(beta_tensor),rev=True)

        return x
    
    def parameters(self):
        """
        Return the learnable parameters of the wrapper
        """

        if self.process_beta_mode == "learnable":
            return list(self.inn.parameters()) + list(self.beta_processing_function.parameters())
        else:
            return self.inn.parameters()

##############################################################
# Wrapper for INN operations with multiple external parameters
##############################################################

class INN_Model__MultipleExternalParameters():
    def __init__(self,
                 d:int,
                 inn:Ff.InvertibleModule,
                 device:str,
                 process_beta_mode:str = "log_beta",
                 embedding_model:Callable=None
                 )->None:
        """
        This is a wrapper class to handle all the operations on the distribution defined by the INN.

        parameters:
            d:                  Dimensionality fo the data space
            inn:                Invertible function
            device:             Device to run the code on 
            process_beta_mode:  How to preprocess the inverse temperature passed to the INN
            embedding_model:    Function mapping the beta_tensor to a d'dimensional embedding
        """

        self.inn = inn
        self.device = device
        self.p_0 = force_to(D.MultivariateNormal(torch.zeros(d).to(device), torch.eye(d)),device)
        self.d = d
        self.process_beta_mode = process_beta_mode

        # Use learneable processing of the condition
        if self.process_beta_mode == "learnable": 
            print("Learable temperature embedding")
            self.beta_processing_function = embedding_model

        else:
            "Use standard temperature embedding"
            self.beta_processing_function = _beta_processing_dict[self.process_beta_mode]

    def load_state_dict(self,path:str)->None:
        """
        Load stored parameters.

        parameters:
            path:   Location of the stored parameters.
        """

        state_dict = torch.load(path)["state_dict"]

        print("Load state dict for invertible function")

        self.inn.load_state_dict(state_dict=state_dict["INN"])

        if self.process_beta_mode == "learnable": 
            print("Load state dict for embedding model")
            self.beta_processing_function.load_state_dict(state_dict=state_dict["Embedder"])

    def eval(self):
        """
        Set model to evaluation mode.
        """
                
        self.inn.eval()

        if self.process_beta_mode == "learnable":
            self.beta_processing_function.eval()

    def train(self,b:bool=True):
        """
        Set model to training mode.
        """
                
        self.inn.train(b)

        if self.process_beta_mode == "learnable":
            self.beta_processing_function.train(b)
    
    def log_prob_p_0(self,z_tensor:torch.tensor)->torch.tensor:
        """
        Compute the log-likelihood of the latent distribution.

        parameters:
            z_tensor:       Latent code

        return:
            log_p_z:        Log-likelihood of the latent code 
        """

        log_p_z = self.p_0.log_prob(z_tensor)
        
        return log_p_z

    def _beta_processing(self,parameter_list:List[Union[torch.tensor,float]])->torch.tensor:
        """
        Compute the condition of the INN

        parameters:
            parameter_list:         List with condition batches for the different external parameters.

        return:
            beta_tensor_processed:  Transformed conditions, combined into one tensor
        """

        # Merge the individual parameters into one tensor
        parameter_tensor = torch.hstack(parameter_list)
        parameter_tensor_processed = self.beta_processing_function(parameter_tensor)

        if parameter_tensor_processed is not None:
            return [parameter_tensor_processed]
        else:
            return parameter_tensor_processed

    def log_prob(self,x:torch.tensor,parameter_list:List[Union[torch.tensor,float]])->torch.tensor:
        """
        Compute the log-likelihood of data points.

        parameters:
            x:                      Data points to evaluate
            parameter_list:         List with condition batches for the different external parameters.

        return:
            log_prob_x:             log-likelihood of x
        """
        
        # If only a float is given for a parameter, use it for the whole batch
        for i in range(len(parameter_list)):
            if isinstance(parameter_list[i],float):
                parameter_list[i] = parameter_list[i] * torch.ones(len(x),1).to(self.device)

        z,jac = self.inn(x,self._beta_processing(parameter_list),rev=False)

        p_0_z = self.log_prob_p_0(z_tensor = z)
        
        log_prob_x =  p_0_z + jac

        return log_prob_x

    def sample(self,n_samples:int,parameter_list:List[torch.Tensor|float])-> torch.tensor:
        """
        Generate Samples following the distribution defined by the INN.

        parameters:
            n_samples:          Number of samples to generate
            parameter_list:     List with condition batches for the different external parameters.

        return:
            x:                  Samples following the distribution defined by the INN
        """
        
        # If only a float is given for a parameter, use it for the whole batch
        for i in range(len(parameter_list)):
            if isinstance(parameter_list[i],float):
                parameter_list[i] = parameter_list[i] * torch.ones(n_samples,1).to(self.device)

        z = self.p_0.sample([n_samples])
        
        x,_ = self.inn(z,self._beta_processing(parameter_list = parameter_list),rev=True)

        return x
    
    def parameters(self):
        """
        return the learnable parameters in the Wrapper
        """

        if self.process_beta_mode == "learnable":
            return list(self.inn.parameters()) + list(self.beta_processing_function.parameters())
        else:
            return self.inn.parameters()
