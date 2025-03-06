import torch
from typing import Callable,Dict,List
import numpy as np
from pinf.models.INN import INN_Model
from pinf.losses.utils import get_loss_from_residuals,get_beta

#####################################################################
# Multiple external conditions
#####################################################################

class Objective_PINF_local_Ground_Truth_Multiple_Parameters():
    def __init__(self,
                 t_burn_in:int,
                 t_full:int,
                 parameter_limit_list:List[List[float]],
                 S:Callable,
                 S_kwargs:List[Dict],
                 dSdparam_list:List[Callable],
                 dSdparam_kwargs_list:List[Dict],
                 n_samples_expectation_approx:int,
                 n_samples_evaluation_per_param:int,
                 n_evaluation_params:int,
                 param_sampler_mode_list:List[str],
                 include_base_params:bool,
                 sample_param_params_list:List[Dict],
                 base_parameter_list:List[List[float]],
                 device:str,
                 residual_processing_parameters:Dict,
                 **kwargs
                 )->None:
        
        """
        parameters:
            t_burn_in:                      The time step after which the burn in phase is finished
            t_full:                         The time step after which the full range is reached
            parameter_limit_list:           List of lists describing the limits along each axis in the parameter space
            S:                              The ground truth function in the exponent of the Gibbs distribution
            S_kwargs:                       Additional parameters for the ground truth energy function
            dSdparam_list:                  List containing the derivatives of the energy function with respect to the different conditions
            dSdparam_kwargs_list:           Additional parameters for the derivatives of the ground truth energy function
            n_samples_expectation_approx:   Number of samples used to approximate the expectation value for each condition value
            n_samples_evaluation_per_param: Number of evaluataion points per condition value.
            n_evaluation_params:            Total number of evaluated condition values in each training step.
            param_sampler_mode_list:        List of sampling modes for the different conditions
            include_base_params:            If True, the loss is computed at the base parameters all the time, in addition to the randomly sampled parameters
            sample_param_params_list:       Additional parameters for the sampling of the conditions
            base_parameter_list:            Points in data space where data is available
            device:                         The device on which the computation is performed
            residual_processing_parameters: Parametersto compute the loss from the residuals
            param_sampler_mode:             Set the method used to sample condition values
        """

        print("*********************************************************************************************")
        print("Use class 'Objective_PINF_local_Ground_Truth_Multiple_Parameters'")
        print("*********************************************************************************************")
        
        # Store settings
        self.device = device
        self.residual_processing_parameters = residual_processing_parameters
        self.include_base_params = include_base_params
        self.t_full = t_full
        self.t_burn_in = t_burn_in

        self.iter_start = None

        self.n_samples_expectation_approx = n_samples_expectation_approx
        self.n_samples_evaluation_per_param = n_samples_evaluation_per_param
        self.n_evaluation_params = n_evaluation_params
        self.param_sampler_mode_list = param_sampler_mode_list

        self.sample_param_params_list = sample_param_params_list

        for i in range(len(self.sample_param_params_list)):
            self.sample_param_params_list[i]["t_full"] = self.t_full
            self.sample_param_params_list[i]["t_burn_in"] = self.t_burn_in

        self.S = S
        self.dSdparam_list = dSdparam_list

        self.S_kwargs = S_kwargs
        self.dSdparam_kwargs = dSdparam_kwargs_list

        # Store the limits of the intervals evaluated for the external parameters
        self.parameter_limit_list = parameter_limit_list

        # Store the base parameters
        self.base_parameter_tensor = torch.Tensor(base_parameter_list)

        # Counter for the number of calls of the loss function
        self.iteration = 0
            
    def __get_loss(self,INN:INN_Model,param_batch:torch.Tensor,EX_batch:torch.tensor,get_eval_points:Callable,directions:torch.Tensor)->torch.Tensor:
        """
        Perform the actual loss computation

        parameters:
            INN:                The INN model
            param_batch:        The batch of condition values at which the loss is evaluated
            EX_batch:           Batch of expectation values for the evaluated conditions
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            directions:         Batch of integers specifying the dimension along which the gradient is computed

        returns:
            residuals:          Residuals between the target and the model gradient for each evaluated point
        """

        # Check inputs
        assert(len(param_batch.shape) == 2)
        assert(EX_batch.shape == torch.Size([len(directions)]))

        # Get the parameter parameter_list
        parameter_list = []

        for i in range(len(self.dSdparam_list)):

            params_i = param_batch[:,i].reshape(-1,1)
            param_tensor_i = torch.ones([len(directions),self.n_samples_evaluation_per_param]) * params_i

            assert (param_tensor_i.shape == torch.Size([len(directions),self.n_samples_evaluation_per_param]))

            param_tensor_flat_i = param_tensor_i.reshape(-1,1).to(self.device)

            parameter_list.append(param_tensor_flat_i)

        # Get expectation values
        EX_batch = EX_batch.reshape(-1,1)
        EX_tensor = torch.ones([len(directions),self.n_samples_evaluation_per_param]) * EX_batch
        assert (EX_tensor.shape == torch.Size([len(directions),self.n_samples_evaluation_per_param]))
        EX_tensor_flat = EX_tensor.reshape(-1).to(self.device)

        directions_tensor = (torch.ones([len(directions),self.n_samples_evaluation_per_param]) * directions.reshape(-1,1)).reshape(-1).to(self.device)

        # Get the target for the gradient
        with torch.no_grad():
            INN.train(False)

            x_eval = get_eval_points(parameter_list = parameter_list)

            A_eval = torch.zeros([len(directions) * self.n_samples_evaluation_per_param]).to(self.device)

            for i in range(len(self.dSdparam_list)):
                A_eval_i = self.dSdparam_list[i](x_eval,parameter_list = parameter_list,**self.dSdparam_kwargs[i]).reshape(-1)
                A_eval += A_eval_i * (directions_tensor == i).float()

            target = EX_tensor_flat - A_eval

            INN.train(True)

        # Compute the gradient of the log-likelihood with respect to the condition along the specified direction
            
        grad = torch.zeros([len(directions) * self.n_samples_evaluation_per_param]).to(self.device)

        for i in range(len(self.dSdparam_list)):    

            parameter_list[i].requires_grad_(True)
            log_p_x_eval_i = INN.log_prob(x_eval,parameter_list = parameter_list)
            grad_i = torch.autograd.grad(log_p_x_eval_i.sum(),parameter_list[i],create_graph=True)[0].squeeze()

            mask = (directions_tensor == i)

            grad[mask] = grad_i[mask]

        assert(grad.shape == target.shape)

        residuals = grad - target.detach()

        assert(residuals.shape == torch.Size([len(directions) * self.n_samples_evaluation_per_param]))

        return residuals
    
    def __sample_param_batch(self)->torch.Tensor:
        """
        Sample a batch of condition point where the loss is evaluated

        returns:
            param_batch:    Batch of condition values
        """

        # In case of burn-in phase or always include base parameters add them first
        if (self.iteration <= self.t_burn_in) or self.include_base_params:
            idx = torch.randperm(len(self.base_parameter_tensor))
            param_batch = self.base_parameter_tensor[idx][:min(len(self.base_parameter_tensor),self.n_evaluation_params)]

        else:
            param_batch = torch.zeros(0)

        # If the maximum number of points is already reached return the batch of parameter values
        if (self.iteration <= self.t_burn_in) or (len(param_batch) == self.n_evaluation_params):
            return param_batch


        n_params_to_sample = int(self.n_evaluation_params - len(param_batch))

        assert(n_params_to_sample > 0)

        for i in range(n_params_to_sample):

            # Get a base parameter at random
            idx = np.random.randint(low = 0,high = len(self.base_parameter_tensor))
            param_star_i = self.base_parameter_tensor[idx]

            param_i = torch.zeros([1,len(param_star_i)])

            # For each coordinate of the parameter point sample a value
            for j in range(len(param_star_i)):
            
                if self.param_sampler_mode_list[j] == "simple":

                    param_i[0][j],left,right = get_beta(
                            t = self.iteration,
                            beta_star=param_star_i[j].item(),
                            beta_min=self.parameter_limit_list[j][0],
                            beta_max=self.parameter_limit_list[j][1],
                            **self.sample_param_params_list[j]
                            )
            
                else:
                    raise NotImplementedError()
                
            
            param_batch = torch.cat((param_i,param_batch),0)

        assert(param_batch.shape == torch.Size([self.n_evaluation_params,len(self.dSdparam_list)]))

        return param_batch

    def __get_expectation_values(self,param_batch:torch.Tensor,INN:INN_Model,directions:torch.Tensor)->torch.Tensor:
        """
        Compute expectation values at the given condition values using self normalized importance sampling.

        parameters:
            param_batch:    Batch of condition values for which the expectation values are approximated.
            INN:            Current INN
            directions:     Batch of integers specifying the dimension along which the gradient is computed

        returns:
            EX_A:           Batch of expectation values
        """

        assert(param_batch.shape == torch.Size([self.n_evaluation_params,len(self.dSdparam_list)]))

        # Get the parameter parameter_list
        parameter_list = []

        for i in range(len(self.dSdparam_list)):

            params_i = param_batch[:,i].reshape(-1,1)
            param_tensor_i = torch.ones([self.n_evaluation_params,self.n_samples_expectation_approx]) * params_i

            assert (param_tensor_i.shape == torch.Size([self.n_evaluation_params,self.n_samples_expectation_approx]))

            param_tensor_flat_i = param_tensor_i.reshape(-1,1).to(self.device)

            parameter_list.append(param_tensor_flat_i)

        directions_tensor = (torch.ones([self.n_evaluation_params,self.n_samples_expectation_approx]) * directions.reshape(-1,1)).reshape(-1).to(self.device)

        # Approximate the expectation value for the given parameter
        with torch.no_grad():
            INN.train(False)

            #1) Get samples from the INN
            x_proposal = INN.sample(n_samples = len(parameter_list[0]),parameter_list = parameter_list)

            #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points along the specified direction
            A_proposal = torch.zeros([self.n_evaluation_params * self.n_samples_expectation_approx]).to(self.device)

            for i in range(len(self.dSdparam_list)):
                A_proposal_i = self.dSdparam_list[i](x_proposal,parameter_list = parameter_list,**self.dSdparam_kwargs[i]).reshape(-1)
                A_proposal += A_proposal_i * (directions_tensor == i).float()

            A_proposal = A_proposal.reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

            #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
            log_p_x_proposal_INN    = INN.log_prob(x_proposal,parameter_list = parameter_list)
            log_p_x_proposal_GT     = - self.S(x_proposal,parameter_list = parameter_list,**self.S_kwargs)

            #4) compute the log likelihood ratios
            log_w = log_p_x_proposal_GT - log_p_x_proposal_INN

            # reshape 
            log_w = log_w.reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

            #5) compute the log parition function
            log_Z = torch.logsumexp(log_w,dim = 1,keepdim=True) - np.log(self.n_samples_expectation_approx)

            assert(log_Z.shape == torch.Size([self.n_evaluation_params,1]))

            #6) Compute the importance weights
            log_omega = log_w - log_Z

            assert(log_omega.shape == A_proposal.shape)

            #7) Compute the sample based expectation value of the energy
            EX_A = (A_proposal * log_omega.exp()).mean(-1)

        assert(EX_A.shape == torch.Size([self.n_evaluation_params]))

        INN.train(True)

        return EX_A.detach().cpu()

    def __call__(self,INN,epoch,get_eval_points,logger = None)->torch.Tensor:
        """
        Compute the physics-informed loss

        parameters:
            INN:                The INN model
            epoch:              The current epoch
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            logger:             The logger for the loss

        returns:
            loss:               The temperature scaling loss
        """

        if self.iter_start is None:
            self.iter_start = self.iteration

        # Sample parameters at which the loss is evaluated
        parameter_batch = self.__sample_param_batch()

        # For each parameter randomly select the direction of the gradient
        directions = torch.randint(0,len(self.dSdparam_list),(len(parameter_batch),))

        # Approximate the Expectation values along the selected direction
        EX_batch = self.__get_expectation_values(param_batch = parameter_batch,INN = INN,directions = directions)

        # Compute the loss
        residuals = self.__get_loss(INN = INN,param_batch=parameter_batch,EX_batch = EX_batch,get_eval_points=get_eval_points,directions = directions)

        # Copute the loss from the residuals
        loss = get_loss_from_residuals(residuals,**self.residual_processing_parameters)

        return loss

#####################################################################
# One external condition
#####################################################################

class Objective_PINF_local_Ground_Truth_one_param_V2():
    def __init__(self,
                 t_burn_in:int,
                 t_full:int,
                 param_min:float,
                 param_max:float,
                 S:Callable,
                 S_kwargs:Dict,
                 dSdparam:Callable,
                 dSdparam_kwargs:Dict,
                 n_samples_expectation_approx:int,
                 n_samples_evaluation_per_param:int,
                 n_evaluation_params:int,
                 param_sampler_mode:str,
                 include_base_params:bool,
                 sample_param_params:Dict,
                 base_params:torch.tensor,
                 device:str,
                 residual_processing_parameters:Dict,
                 n_bins_storage:int = 100,
                 **kwargs
                 )->None:

        """
        parameters:
            t_burn_in:                      The time step after which the burn in phase is finished
            t_full:                         The time step after which the full range is reached
            param_min:                      The minimal parameter value
            param_max:                      The maximal parameter value
            S:                              The ground truth function in the exponent of the Gibbs distribution
            S_kwargs:                       Additional parameters for the ground truth energy function
            dSdparam:                       The derivative of the ground truth function with respect to the parameter
            dSdparam_kwargs:                Additional parameters for the derivative of the ground truth energy function
            n_samples_expectation_approx:   Number of samples used to approximate the expectation value for each condition value
            n_samples_evaluation_per_param: Number of evaluataion points per condition value.
            n_evaluation_params:            Total number of evaluated condition values in each training step.
            include_base_params:            If True, the loss is computed at the base parameters all the time, in addition to the randomly sampled parameters
            sample_param_params:            Parameters for the sampling of the parameter at which the loss is computed
            base_params:                    The base parameters at which the nll loss is computed
            device:                         The device on which the computation is performed
            residual_processing_parameters: Parametersto compute the loss from the residuals
            param_sampler_mode:             Set the method used to sample condition values
            n_bins_storage:                 Number of bins for storing the approximated expectation values (for visualization only)
        """

        print("*********************************************************************************************")
        print("Use class 'Objective_PINF_local_Ground_Truth_one_param_V2'")
        print("*********************************************************************************************")
        
        # Store settings
        self.device = device
        self.residual_processing_parameters = residual_processing_parameters
        self.include_base_params = include_base_params
        self.t_full = t_full
        self.t_burn_in = t_burn_in

        self.n_samples_expectation_approx = n_samples_expectation_approx
        self.n_samples_evaluation_per_param = n_samples_evaluation_per_param
        self.n_evaluation_params = n_evaluation_params
        self.param_sampler_mode = param_sampler_mode

        self.sample_param_params = sample_param_params
        self.sample_param_params["t_full"] = self.t_full
        self.sample_param_params["t_burn_in"] = self.t_burn_in

        self.S = S
        self.dSdparam = dSdparam
        self.S_kwargs = S_kwargs
        self.dSdparam_kwargs = dSdparam_kwargs

        # Set the limits of the beta ranges for the individual base temperatures
        global_min = torch.ones([len(base_params),1]) * param_min
        global_max = torch.ones([len(base_params),1]) * param_max

        selected_min = global_min
        selected_max = global_max

        self.params_mins = []
        self.params_maxs = []
        self.params_stars = []

        for i in range(len(base_params)):
            self.params_mins.append(selected_min[i].item())
            self.params_maxs.append(selected_max[i].item())
            self.params_stars.append(base_params[i].item())

        #Store the approximated expectation vlues (only for evaluation and visualization)

        # Initialize the bins of the grid
        self.param_bin_edges = torch.linspace(np.log(param_min),np.log(param_max),n_bins_storage+1).exp()
        self.param_storage_grid = torch.zeros(n_bins_storage)
        self.EX_storage_grid = torch.zeros(n_bins_storage)

        # Counter for the number of calls of the loss function
        self.iteration = 0

    def get_loss(self,INN:INN_Model,param_batch:torch.Tensor,EX_batch:torch.tensor,get_eval_points:Callable)->torch.Tensor:
        """
        Perform the actual loss computation

        parameters:
            INN:                The INN model
            param_batch:        The batch of condition values at which the loss is evaluated
            EX_batch:           Batch of expectation values for the evaluated conditions
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor

        returns:
            loss:               The physics-informed loss
        """

        # Check inputs
        assert(len(param_batch.shape) == 1)
        assert(EX_batch.shape == torch.Size([self.n_evaluation_params]))

        # Get condition values
        param_batch = param_batch.reshape(-1,1)
        param_tensor = torch.ones([self.n_evaluation_params,self.n_samples_evaluation_per_param]) * param_batch

        assert (param_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_evaluation_per_param]))

        param_tensor_flat = param_tensor.reshape(-1,1).to(self.device)

        # Get expectation values
        EX_batch = EX_batch.reshape(-1,1)
        EX_tensor = torch.ones([self.n_evaluation_params,self.n_samples_evaluation_per_param]) * EX_batch

        assert (EX_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_evaluation_per_param]))

        EX_tensor_flat = EX_tensor.reshape(-1).to(self.device)

        # Get the target for the gradient
        with torch.no_grad():
            INN.train(False)

            x_eval = get_eval_points(beta_tensor = param_tensor_flat)

            #10) Compute the ground truth energies of the evaluation points
            A_eval = self.dSdparam(x_eval,**self.dSdparam_kwargs)

            assert(EX_tensor_flat.shape == A_eval.shape)

            #11) Compute the target
            target = EX_tensor_flat - A_eval
            
            INN.train(True)

        # Compute the gradient of the log-likelihood with respect to the condition
        param_tensor_flat.requires_grad_(True)

        log_p_x_eval = INN.log_prob(x_eval,param_tensor_flat)

        grad = torch.autograd.grad(log_p_x_eval.sum(),param_tensor_flat,create_graph=True)[0].squeeze()

        # Compute the residuals
        assert(grad.shape == target.shape)

        residuals = grad - target.detach()
        assert(residuals.shape == torch.Size([self.n_evaluation_params * self.n_samples_evaluation_per_param]))

        # Compute the loss from the residuals
        loss = get_loss_from_residuals(residuals,**self.residual_processing_parameters)

        return loss

    def __sample_param_batch(self)->torch.Tensor:
        """
        Sample a batch of condition values where the loss is evaluated

        returns:
            param_batch:    Batch of condition values
        """
        
        # In case of burn-in phase or always include base parameters add them first
        if (self.iteration <= self.t_burn_in) or self.include_base_params:

            param_batch = torch.tensor(self.params_stars)
            idx = torch.randperm(len(self.params_stars))
            param_batch = param_batch[idx][:min(len(self.params_stars),self.n_evaluation_params)]

        else:
            param_batch = torch.zeros(0)

        # If the maximum number of points is already reached return the batch of parameter values
        if (self.iteration <= self.t_burn_in) or (len(param_batch) == self.n_evaluation_params):
            return param_batch
        
        n_params_to_sample = int(self.n_evaluation_params - len(param_batch))

        assert(n_params_to_sample > 0)

        for i in range(n_params_to_sample):
            
            # Sample condition values
            if self.param_sampler_mode == "simple":
                # Get a base temperature at random
                idx = np.random.randint(low = 0,high = len(self.params_stars))
                param_star_i = self.params_stars[idx]

                param_i,left,right = get_beta(
                        t = self.iteration,
                        beta_star=param_star_i,
                        beta_min=self.params_mins[idx],
                        beta_max=self.params_maxs[idx],
                        **self.sample_param_params
                        )
                
                param_batch = torch.cat((torch.Tensor([param_i]),param_batch),0)

            else:
                raise NotImplementedError()

        assert(param_batch.shape == torch.Size([self.n_evaluation_params]))

        return param_batch

    def get_expectation_values(self,param_batch:torch.Tensor,INN:INN_Model)->torch.Tensor:
        """
        Compute expectation values at the given condition values using self normalized importance sampling.

        parameters:
            param_batch:    Batch of condition values for which the expectation values are approximated.
            INN:            Current INN

        returns:
            EX_A:           Batch of expectation values
        """

        assert(len(param_batch.shape) == 1)

        # Get one large parameter tensor
        param_batch = param_batch.reshape(-1,1)
        param_tensor = torch.ones([self.n_evaluation_params,self.n_samples_expectation_approx]) * param_batch

        assert (param_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_expectation_approx]))
        
        param_tensor_flat = param_tensor.reshape(-1,1).to(self.device)
        
        # Approximate the expectation value for the given parameter
        with torch.no_grad():
            INN.train(False)

            #1) Get samples from the INN
            x_proposal = INN.sample(n_samples = len(param_tensor_flat),beta_tensor = param_tensor_flat)

            #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points
            A_proposal = self.dSdparam(x_proposal,**self.dSdparam_kwargs).reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

            #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
            log_p_x_proposal_INN    = INN.log_prob(x_proposal,param_tensor_flat)
            log_p_x_proposal_GT     = - self.S(x_proposal,param_tensor_flat,**self.S_kwargs)

            assert(log_p_x_proposal_INN.shape == log_p_x_proposal_GT.shape)

            #4) compute the log likelihood ratios
            log_w = log_p_x_proposal_GT - log_p_x_proposal_INN

            # reshape 
            log_w = log_w.reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

            #5) compute the log parition function
            log_Z = torch.logsumexp(log_w,dim = 1,keepdim=True) - np.log(self.n_samples_expectation_approx)

            assert(log_Z.shape == torch.Size([self.n_evaluation_params,1]))

            #6) Compute the importance weights
            log_omega = log_w - log_Z

            assert(log_omega.shape == A_proposal.shape)

            #7) Compute the sample based expectation value of the energy
            EX_A = (A_proposal * log_omega.exp()).mean(-1)
            
            INN.train(True)

        assert(EX_A.shape == torch.Size([self.n_evaluation_params]))

        return EX_A.detach().cpu()

    def __call__(self,INN,epoch,get_eval_points,logger = None)->torch.Tensor:
        """
        Compute the physics-informed loss contribution

        parameters:
            INN:                The INN model
            epoch:              The current epoch
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            logger:             The logger for the loss

        returns:
            loss:               The temperature scaling loss
        """

        # Get a batch of parameters at which the loss is evluated
        param_batch = self.__sample_param_batch()

        # Get the expectation values for the given batch of parameters
        EX_batch = self.get_expectation_values(param_batch=param_batch,INN = INN)

        loss = self.get_loss(INN = INN,param_batch=param_batch,EX_batch = EX_batch,get_eval_points=get_eval_points)

        logger.experiment.add_scalar(f"metadata/loss_model_internal_iteratoins",self.iteration,self.iteration)

        # For evaluation only
        # Store the approximated expectation values for the given parameter batch
        bin_idx = torch.searchsorted(self.param_bin_edges, param_batch, right=False) - 1

        # If some indices are there multiple times only store one
        unique, idx, counts = torch.unique(bin_idx, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]

        unique_bin_indices = bin_idx[first_indicies]
    
        self.param_storage_grid[unique_bin_indices] = param_batch.squeeze()[first_indicies]
        self.EX_storage_grid[unique_bin_indices] = EX_batch.squeeze()[first_indicies]

        return loss


class Objective_PINF_local_Ground_Truth_one_param_V3(Objective_PINF_local_Ground_Truth_one_param_V2):
    def __init__(self,
                 t_burn_in:int,
                 t_full:int,
                 param_min:float,
                 param_max:float,
                 S:Callable,
                 S_kwargs:Dict,
                 dSdparam:Callable,
                 dSdparam_kwargs:Dict,
                 n_samples_expectation_approx:int,
                 n_samples_evaluation_per_param:int,
                 n_evaluation_params:int,
                 param_sampler_mode:str,
                 include_base_params:bool,
                 sample_param_params:Dict,
                 base_params:torch.tensor,
                 device:str,
                 residual_processing_parameters:Dict,
                 n_bins_storage:int = 100,
                 use_learned_energy:bool = False,
                 **kwargs
                 )->None:
        
        """
        Adaption of 'Objective_PINF_local_Ground_Truth_one_param_V2', allow usage of learned distribution to approximate the target distribution in 
        the case of power-scaling.

        parameters:
            t_burn_in:                      The time step after which the burn in phase is finished
            t_full:                         The time step after which the full range is reached
            param_min:                      The minimal parameter value
            param_max:                      The maximal parameter value
            S:                              The ground truth function in the exponent of the Gibbs distribution
            S_kwargs:                       Additional parameters for the ground truth energy function
            dSdparam:                       The derivative of the ground truth function with respect to the parameter
            dSdparam_kwargs:                Additional parameters for the derivative of the ground truth energy function
            n_samples_expectation_approx:   Number of samples used to approximate the expectation value for each condition value
            n_samples_evaluation_per_param: Number of evaluataion points per condition value.
            n_evaluation_params:            Total number of evaluated condition values in each training step.
            include_base_params:            If True, the loss is computed at the base parameters all the time, in addition to the randomly sampled parameters
            sample_param_params:            Parameters for the sampling of the parameter at which the loss is computed
            base_params:                    The base parameters at which the nll loss is computed
            device:                         The device on which the computation is performed
            residual_processing_parameters: Parametersto compute the loss from the residuals
            param_sampler_mode:             Set the method used to sample condition values
            n_bins_storage:                 Number of bins for storing the approximated expectation values (for visualization only)
            use_learned_energy:             Set to true to allow usage of the learned energy to approximate the target distribution
        """
        
        super().__init__(
                 t_burn_in = t_burn_in,
                 t_full = t_full,
                 param_min = param_min,
                 param_max = param_max,
                 S = S,
                 S_kwargs = S_kwargs,
                 dSdparam = dSdparam,
                 dSdparam_kwargs = dSdparam_kwargs,
                 n_samples_expectation_approx = n_samples_expectation_approx,
                 n_samples_evaluation_per_param = n_samples_evaluation_per_param,
                 n_evaluation_params = n_evaluation_params,
                 param_sampler_mode = param_sampler_mode,
                 include_base_params = include_base_params,
                 sample_param_params = sample_param_params,
                 base_params = base_params,
                 device = device,
                 residual_processing_parameters = residual_processing_parameters,
                 n_bins_storage = n_bins_storage,
                 **kwargs)
        
        print("*********************************************************************************************")
        print("Use class 'Objective_PINF_local_Ground_Truth_one_param_V3'")
        print("*********************************************************************************************")

        self.use_learned_energy = use_learned_energy

        if self.use_learned_energy:
            print("#######################################################################\n")
            print("WARNING:\n")
            print("Learned density is used for training. Only applicable for power-scaling!\n")
            print("#######################################################################\n")

            assert(len(self.params_stars) == 1)

            self.S = None
            self.dSdparam = None
            self.dSdparam_kwargs = None
            self.S_kwargs = None

    def get_loss(self,INN:INN_Model,param_batch:torch.Tensor,EX_batch:torch.tensor,get_eval_points:Callable)->torch.Tensor:
        """
        Perform the actual loss computation

        parameters:
            INN:                The INN model
            param_batch:        The batch of condition values at which the loss is evaluated
            EX_batch:           Batch of expectation values for the evaluated conditions
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor

        returns:
            loss:               The physics-informed loss
        """

        # Check inputs
        assert(len(param_batch.shape) == 1)
        assert(EX_batch.shape == torch.Size([self.n_evaluation_params]))

        # Get condition values
        param_batch = param_batch.reshape(-1,1)
        param_tensor = torch.ones([self.n_evaluation_params,self.n_samples_evaluation_per_param]) * param_batch

        assert (param_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_evaluation_per_param]))

        param_tensor_flat = param_tensor.reshape(-1,1).to(self.device)

        # Get expectation values
        EX_batch = EX_batch.reshape(-1,1)
        EX_tensor = torch.ones([self.n_evaluation_params,self.n_samples_evaluation_per_param]) * EX_batch

        assert (EX_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_evaluation_per_param]))

        EX_tensor_flat = EX_tensor.reshape(-1).to(self.device)

        # Get the target for the gradient
        with torch.no_grad():
            INN.train(False)

            x_eval = get_eval_points(beta_tensor = param_tensor_flat)

            #10) Compute the ground truth energies of the evaluation points

            #Use learned energy
            if self.use_learned_energy:
                A_eval = - 1 / self.params_stars[0] * INN.log_prob(x_eval,self.params_stars[0]).detach()
            
            #Use ground truth energy
            else:
                A_eval = self.dSdparam(x_eval,**self.dSdparam_kwargs)

            assert(EX_tensor_flat.shape == A_eval.shape)

            #11) Compute the target
            target = EX_tensor_flat - A_eval
            
            INN.train(True)

        # Compute the gradient of the log-likelihood with respect to the condition
        param_tensor_flat.requires_grad_(True)

        log_p_x_eval = INN.log_prob(x_eval,param_tensor_flat)

        grad = torch.autograd.grad(log_p_x_eval.sum(),param_tensor_flat,create_graph=True)[0].squeeze()

        # Compute the residuals
        assert(grad.shape == target.shape)

        residuals = grad - target.detach()
        assert(residuals.shape == torch.Size([self.n_evaluation_params * self.n_samples_evaluation_per_param]))

        # Compute the loss from the residuals
        loss = get_loss_from_residuals(residuals,**self.residual_processing_parameters)

        return loss

    def get_expectation_values(self,param_batch:torch.Tensor,INN:INN_Model)->torch.Tensor:
        """
        Compute expectation values at the given condition values using self normalized importance sampling.

        parameters:
            param_batch:    Batch of condition values for which the expectation values are approximated.
            INN:            Current INN

        returns:
            EX_A:           Batch of expectation values
        """

        assert(len(param_batch.shape) == 1)

        # Get one large parameter tensor
        param_batch = param_batch.reshape(-1,1)
        param_tensor = torch.ones([self.n_evaluation_params,self.n_samples_expectation_approx]) * param_batch
        assert (param_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_expectation_approx]))
        
        param_tensor_flat = param_tensor.reshape(-1,1).to(self.device)
        
        # Approximate the expectation value for the given parameter
        with torch.no_grad():
            INN.train(False)

            #1) Get samples from the INN
            x_proposal = INN.sample(n_samples = len(param_tensor_flat),beta_tensor = param_tensor_flat)

            #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points

            # Use learned energy
            if self.use_learned_energy:
                A_proposal = - 1 / self.params_stars[0] * INN.log_prob(x_proposal,self.params_stars[0]).detach().reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

                #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
                log_p_x_proposal_GT     = param_tensor_flat.squeeze() / self.params_stars[0] * INN.log_prob(x_proposal,self.params_stars[0]).detach()

            # Use ground truth
            else:
                A_proposal = self.dSdparam(x_proposal,**self.dSdparam_kwargs).reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

                #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
                
                log_p_x_proposal_GT     = - self.S(x_proposal,param_tensor_flat,**self.S_kwargs)

            log_p_x_proposal_INN    = INN.log_prob(x_proposal,param_tensor_flat)

            assert(log_p_x_proposal_INN.shape == log_p_x_proposal_GT.shape)

            #4) compute the log likelihood ratios
            log_w = log_p_x_proposal_GT - log_p_x_proposal_INN

            # reshape 
            log_w = log_w.reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

            #5) compute the log parition function
            log_Z = torch.logsumexp(log_w,dim = 1,keepdim=True) - np.log(self.n_samples_expectation_approx)

            assert(log_Z.shape == torch.Size([self.n_evaluation_params,1]))

            #6) Compute the importance weights
            log_omega = log_w - log_Z

            assert(log_omega.shape == A_proposal.shape)

            #7) Compute the sample based expectation value of the energy
            EX_A = (A_proposal * log_omega.exp()).mean(-1)
            
            INN.train(True)

        assert(EX_A.shape == torch.Size([self.n_evaluation_params]))

        return EX_A.detach().cpu()
