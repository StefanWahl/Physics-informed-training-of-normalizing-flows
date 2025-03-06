import torch
from typing import Callable,Dict,List
import numpy as np
from pinf.models.INN import INN_Model
from pinf.losses.utils import get_loss_from_residuals,get_beta
import tqdm

#####################################################################
# One external condition
#####################################################################

class Objective_PINF_parallel_Ground_Truth_one_param():
    def __init__(self,
                 param_min:float,
                 param_max:float,
                 S:Callable,
                 S_kwargs:Dict,
                 dSdparam:Callable,
                 dSdparam_kwargs:Dict,
                 n_reps:int,
                 base_params:torch.tensor,
                 device:str,
                 epsilon_causality_weight:float,
                 n_points_param_grid:int,
                 residual_processing_parameters:Dict,
                 alpha_running_EX_A:float,
                 average_importance_weights:bool,
                 update_freq_running_average_EX_A:int,
                 bs:int,
                 alpha_running_loss:float,
                 n_samples_expectation_computation:int,
                 bs_expectation_computation:int,
                 init_param_grid_in_log_space:bool,
                 n_epochs:int,
                 n_batches_per_epoch:int,
                 epsilon_causality_decay_factor:float,
                 epsilon_reduction_factor_inbetween:float
                 )->None:

        """
        parameters:
            param_min:                          The minimal parameter value
            param_max:                          The maximal parameter value
            S:                                  The ground truth function in the exponent of the Gibbs distribution
            S_kwargs:                           Additional parameters for the ground truth energy function
            dSdparam:                           The derivative of the ground truth function with respect to the parameter
            dSdparam_kwargs:                    Additional parameters for the derivative of the ground truth energy function
            n_reps:                             The number of repetitions for the loss computation
            base_params:                        The base parameters at which the nll loss is computed
            device:                             The device on which the computation is performed
            residual_processing_parameters:     Parametersto compute the loss from the residuals
            bs:                                 The batch size for the computation of the loss
            epsilon_causality_weight:           The weight of the causality term in the loss
            n_points_param_grid:int,            The number of points in the parameter grid
            alpha_running_EX_A:                 The running average parameter for the expectation value of the derivate of the ground truth energy function with respect to the parameter
            average_importance_weights:         If True, the importance weights are averaged if they are inbetween of two base parameters
            update_freq_running_average_EX_A:   The frequency with which the running average of the expectation value is updated
            alpha_running_loss:                 The running average parameter for the loss
            n_samples_expectation_computation:  The number of samples used to compute the expectation values
            bs_expectation_computation:         The batch size for the computation of the expectation values
            init_param_grid_in_log_space:       If True, the parameter grid is initialized in log space
            n_epochs:                           Number of epochs in which this loss is used
            n_batches_per_epoch:                Number of optimization steps per epoch
            epsilon_causality_decay_factor:     Final ration of the initial value of epsilon (reduced by applying exponential decay)
            epsilon_reduction_factor_inbetween: Factor by which epsilon is reduced in between two base parameters
        """

        print("*********************************************************************************************")
        print("Use class 'Objective_PINF_parallel_Ground_Truth_one_param'")
        print("*********************************************************************************************")

        T = n_epochs * n_batches_per_epoch
        self.tau_cw = - np.log(epsilon_causality_decay_factor) / T

        # Store settings
        self.device = device
        self.bs = bs
        self.residual_processing_parameters = residual_processing_parameters

        self.S = S
        self.dSdparam = dSdparam
        self.S_kwargs = S_kwargs
        self.dSdparam_kwargs = dSdparam_kwargs
        self.n_reps = n_reps
        self.epsilon_causality_weight = epsilon_causality_weight

        self.update_freq_running_average_EX_A = update_freq_running_average_EX_A
        self.n_samples_expectation_computation = n_samples_expectation_computation
        self.bs_expectation_computation = bs_expectation_computation

        # Initialize the parameter grid
        if init_param_grid_in_log_space:
            log_param_grid = torch.linspace(np.log(param_min),np.log(param_max),n_points_param_grid).reshape(-1,1)
            self.param_grid = torch.exp(log_param_grid)
        
        else:
            self.param_grid = torch.linspace(param_min,param_max,n_points_param_grid).reshape(-1,1)

        self.param_grid = torch.cat((base_params.reshape(-1,1),self.param_grid),0)

        # Remove duplicates
        unique_elements = torch.unique(self.param_grid)
        self.param_grid = unique_elements.reshape(-1,1)

        # Ensure, that the parameters are sorted in ascending order
        self.param_grid,_ = torch.sort(self.param_grid,dim = 0)

        # Get the indices of the base parameters
        self.n_points_param_grid = n_points_param_grid
        self.base_params = base_params

        # Intialize the running average stores for the expectation values
        self.EX_A = None
        self.alpha_running_EX_A = alpha_running_EX_A
        self.average_importance_weights = average_importance_weights

        # Counter for the number of calls of the loss function
        self.iteration = 0

        assert(self.alpha_running_EX_A >= 0.0)
        assert(self.alpha_running_EX_A <= 1.0)
        assert(self.epsilon_causality_weight >= 0.0)

        # Statistics for the loss in the individual bins
        self.loss_statistics = torch.zeros(len(self.param_grid))
        self.alpha_running_loss = alpha_running_loss
        self.freq_update_causality_weights = 50
        self.log_causality_weights = None

        # Reduce epsilon at grid points in between two base parameters
        self.multiplyer = torch.ones_like(self.param_grid)
        
        for i,param in enumerate(self.param_grid):

            # Check if it is inbetween two the base parameters
            mask_1 = (param < self.base_params).sum().item()
            mask_2 = (param > self.base_params).sum().item()

            flag = mask_1 * mask_2

            if flag == 1:
                self.multiplyer[i] = epsilon_reduction_factor_inbetween

    def get_loss(self,INN:INN_Model,get_eval_points:Callable)->torch.Tensor:
        """
        Perform the actual loss computation

        parameters:
            INN:                The INN model
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor.

        returns:
            loss:               The physics-informed loss
        """

        #############################################################################################################
        #1) Get randomly select a batch of parameters for the evaluation
        #############################################################################################################

        # Initial call: Uniformly sample the points
        if self.log_causality_weights is None:
            print("Initial call")
            idx = torch.randint(low = 0,high = len(self.param_grid),size = (self.bs,))

        # Follow up calls: Sample the indices based on the distribution defined by the causality weights 
        else:
            m = torch.distributions.categorical.Categorical(logits = self.log_causality_weights)
            idx = m.sample([self.bs]).cpu()
        
        assert(idx.shape == torch.Size([self.bs]))

        param_tensor = self.param_grid[idx].to(self.device)
        assert(param_tensor.shape == torch.Size([self.bs,1]))

        #############################################################################################################
        #2) Compute the target for the logarithm of the INN distribution
        #############################################################################################################
        with torch.no_grad():

            INN.train(False)

            #2a) Get evaluation points at which the loss is evaluated
            x_eval = get_eval_points(beta_tensor = param_tensor)

            #2b) Compute the ground truth energies of the evaluation points
            A_eval = self.dSdparam(x_eval,**self.dSdparam_kwargs).reshape(-1,1)
            assert(A_eval.shape == param_tensor.shape)

            #2c) Compute the target
            target = (self.EX_A[idx] - A_eval).detach()
            assert(self.EX_A[idx].shape == torch.Size([self.bs,1]))
            assert(target.shape == torch.Size([self.bs,1]))

            INN.train(True)

        #############################################################################################################
        #3) Compute the gradient of the INN ditsribution with respect to the parameter
        #############################################################################################################
        param_tensor.requires_grad_(True)

        #3a) Compute the log likelihood of the evaluation points under the INN distribution
        log_p_x_eval = INN.log_prob(x_eval,param_tensor)

        #3b) Compute the gradient of the log likelihood of the evaluation points under the INN distribution with respect to the parameter
        grad = torch.autograd.grad(log_p_x_eval.sum(),param_tensor,create_graph=True)[0]

        assert(grad.shape == param_tensor.shape)
        assert(grad.shape == target.shape)

        #############################################################################################################
        #4)Compute loss for each of the evaluation points
        #############################################################################################################
        
        #4a) Get the residuals
        residuals = grad - target.detach()
        assert(residuals.shape == param_tensor.shape)

        #4b) Compute the loss based on the residuals
        loss = get_loss_from_residuals(residuals,dim = 1,**self.residual_processing_parameters)
        assert(loss.shape == torch.Size([param_tensor.shape[0]]))

        #############################################################################################################
        #5) Update the running averages of the loss on the evaluated grid points
        #############################################################################################################

        #5a) Average the losses if there are multiple samples for the same parameter
        unique_idx,position_uniques,counts_uniques = torch.unique(idx,return_counts = True,return_inverse = True)

        target_tensor_loss = torch.zeros(unique_idx.shape[0])
        target_tensor_loss.scatter_add_(0, position_uniques, loss.cpu().detach()) / counts_uniques
        target_tensor_loss = target_tensor_loss / counts_uniques

        #5b) Compute weights depending on the number of samples for the same parameter
        alphas_reweighted = self.alpha_running_loss**counts_uniques  

        #5c) Compute running average
        assert(self.loss_statistics[unique_idx].shape == target_tensor_loss.shape)
        assert(self.loss_statistics[unique_idx].shape == alphas_reweighted.shape)

        w_old = alphas_reweighted
        w_new = 1.0 - w_old

        assert(w_old.shape == alphas_reweighted.shape)
        assert(w_new.shape == w_old.shape)

        self.loss_statistics[unique_idx.cpu()] = w_old * self.loss_statistics[unique_idx] + w_new * target_tensor_loss.cpu()
        assert(self.loss_statistics.shape == torch.Size([len(self.param_grid)]))

        #############################################################################################################
        #6) If applicable, update the logarithms of the causality weights
        #############################################################################################################
        if (self.log_causality_weights is None) or ((self.iteration % self.freq_update_causality_weights) == 0):
            self.log_causality_weights = self.compute_causality_weights_exponents(self.loss_statistics.detach(),self.param_grid)

        #############################################################################################################
        #7) Aggregate the loss
        #############################################################################################################
        loss = loss.mean()

        assert(loss.shape == torch.Size([]))

        return loss
    
    def compute_causality_weights_exponents(self,loss:torch.tensor,param_tensor:torch.tensor)->torch.tensor:
        '''
        Compute the logarithms of the causality weights for the individual grid points.

        parameters:
            loss:               Tensor of shape [K] contaiing the loss values at the evaluated parameter values
            param_tensor:       Tensor of shape [K,self.bs] containing the parameter values at which the loss is evaluated. Has to be sorted in ascending order

        returns:
            causality_weights:  Tensor of shape [K] containing the logarithms of the causality weights for the individual losses
        '''

        assert(len(loss.shape) == 1)
        assert(len(param_tensor.shape) == 2)
        assert(loss.shape[0] == param_tensor.shape[0])

        # Compute the epsilon used for the causality weights
        self.epsilon_t = self.epsilon_causality_weight * np.exp(- self.tau_cw * (self.iteration - self.iter_start))

        with torch.no_grad():
            causality_weights = torch.zeros(param_tensor.shape[0]).to(self.device)

            for i in range(len(param_tensor)):

                param_i = param_tensor[i][0]

                # Get the index of the closest base parameter
                a = torch.argmin((self.base_params.cpu().detach() - param_i.cpu().detach()).abs()).item()
                param_base = self.base_params[a]

                idx_base = torch.where(param_tensor[:,0] == param_base)[0].item()
                idx_parameter = i

                # Get the second closest base parameter
                if len(self.base_params) > 1:
                    mask = self.base_params != param_base
                    base_params_masked = self.base_params[mask]

                    b = torch.argmin((base_params_masked.cpu().detach() - param_i.cpu().detach()).abs()).item()
                    param_base_second = base_params_masked[b]
                    idx_base_second = torch.where(param_tensor[:,0] == param_base_second)[0].item()

                # Get the loss weights based on the closest base parameter
                if idx_base < idx_parameter:
                    s1 = loss[idx_base:idx_parameter]

                elif idx_base > idx_parameter:
                    s1 = loss[idx_parameter+1:idx_base+1]

                else:
                    s1 = torch.zeros(1).to(self.device)

                exponent_closest = (-s1.sum() * self.epsilon_t).detach()

                if len(self.base_params) > 1:
                    # If applicable, get the loss weights based on the second closest base parameter, i.e. if the sample is between two base parameters
                    is_between = (param_base < param_i < param_base_second) or (param_base > param_i > param_base_second)

                    if is_between and self.average_importance_weights:
                        # Get the loss weights based on the closest base parameter
                        if idx_base_second < idx_parameter:
                            s2 = loss[idx_base_second:idx_parameter]

                            d_base_idx = idx_parameter - (idx_base_second + 1)

                        elif idx_base_second > idx_parameter:
                            s2 = loss[idx_parameter+1:idx_base_second+1]

                            d_base_idx = idx_base_second - 1 - idx_parameter

                        else:
                            raise ValueError("Not suported case for causality weights")

                        exponent_second = (-s2.sum() * self.epsilon_t).detach()

                        # Get the relative weigting based on the distance to the base temperature
                        d_base_base = abs(idx_base_second - idx_base) - 2

                        assert(d_base_base >= 0)
                        assert(d_base_idx >= 0)
                        assert(d_base_base >= d_base_idx)

                        k = d_base_idx / d_base_base

                        # Catch edge cases where the weights ignore one of the two contributions
                        if k == 1.0:
                            exponent = exponent_closest
                        
                        elif k == 0.0:
                            exponent = exponent_second
                        
                        else:
                            exponent = torch.logsumexp(torch.tensor([exponent_closest + np.log(k),exponent_second + np.log(1-k)]),0)

                        assert(exponent.shape == torch.Size([]))

                    else:
                        exponent = exponent_closest

                else:
                    exponent = exponent_closest
                
                causality_weights[i] = exponent

            causality_weights = causality_weights.detach()

            assert(causality_weights.shape == self.multiplyer.squeeze().shape)

            causality_weights = causality_weights * self.multiplyer.squeeze().to(causality_weights.device)

            return causality_weights
    
    def get_expectation_values(self,INN:INN_Model,n_samples:int)->torch.tensor:
        """
        Compute the expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points

        parameters:
            INN:                The INN model
            n_samples:          The number of samples used to compute the expectation values

        returns:
            EX_A_temporaray:    The approximated expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points
        """

        INN.train(False)
        
        with torch.no_grad():

            EX_A_temporaray = torch.zeros(len(self.param_grid),1).to(self.device)
            n_batches = int(self.n_samples_expectation_computation / self.bs_expectation_computation)

            for i in tqdm.tqdm(range(len(self.param_grid))):

                param = self.param_grid[i]

                log_p_x_proposal_INN = torch.zeros(self.bs_expectation_computation * n_batches).to(self.device)
                log_p_x_proposal_GT = torch.zeros(self.bs_expectation_computation* n_batches).to(self.device)
                A_proposal = torch.zeros(self.bs_expectation_computation* n_batches).to(self.device)

                for j in range(n_batches):

                    #1) Get samples from the INN
                    x_proposal_i = INN.sample(n_samples = self.bs_expectation_computation,beta_tensor = param.item())

                    #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points
                    A_proposal_i = self.dSdparam(x_proposal_i,**self.dSdparam_kwargs)

                    assert(A_proposal_i.shape == torch.Size([self.bs_expectation_computation]))

                    #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
                    log_p_x_proposal_INN_i    = INN.log_prob(x_proposal_i,param.item())
                    log_p_x_proposal_GT_i     = - self.S(x_proposal_i,param.item(),**self.S_kwargs)

                    assert(log_p_x_proposal_GT_i.shape == torch.Size([self.bs_expectation_computation]))
                    assert(log_p_x_proposal_INN_i.shape == torch.Size([self.bs_expectation_computation]))

                    log_p_x_proposal_GT[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_GT_i
                    log_p_x_proposal_INN[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_INN_i
                    A_proposal[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = A_proposal_i

                assert(log_p_x_proposal_INN.shape == log_p_x_proposal_GT.shape)
                assert(log_p_x_proposal_INN.shape == torch.Size([n_samples]))

                #4) compute the log likelihood ratios
                log_w = log_p_x_proposal_GT - log_p_x_proposal_INN

                assert(log_w.shape == torch.Size([n_samples]))

                #5) compute the log parition function
                log_Z = torch.logsumexp(log_w,dim = 0) - np.log(n_samples)

                assert(log_Z.shape == torch.Size([]))

                #6) Compute the importance weights
                log_omega = log_w - log_Z

                assert(log_omega.shape == A_proposal.shape)

                #7) Compute the sample based expectation value of the energy
                EX_A_temporaray[i] = (A_proposal * log_omega.exp()).mean().item()
                
            assert(EX_A_temporaray.shape == torch.Size([len(self.param_grid),1]))
    
            INN.train(True)

            return EX_A_temporaray
        
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

        #############################################################################################################
        #Update the running average store for the expectation values
        #############################################################################################################

        # Initialize the running average store for the expectation values at the first call
        if self.EX_A is None:

            self.iter_start = self.iteration

            print("Initialize running average store for expectation values")
            self.EX_A =  self.get_expectation_values(INN = INN,n_samples = self.n_samples_expectation_computation)
            print("done")

        # Update the running average store for the expectation values
        elif (self.iteration - self.iter_start) % self.update_freq_running_average_EX_A == 0:
            print("Update running average store for expectation values")
            update_EX_A = self.get_expectation_values(INN = INN,n_samples = self.n_samples_expectation_computation)
            self.EX_A = self.alpha_running_EX_A * self.EX_A + (1.0 - self.alpha_running_EX_A) * update_EX_A

        #############################################################################################################
        #Compute the loss
        #############################################################################################################
            
        # Count the number of evaluations
        counter = 0
        loss = torch.zeros(1).to(self.device)

        for i in range(self.n_reps):

            loss_i = self.get_loss(INN = INN,get_eval_points = get_eval_points)
            loss = loss + loss_i
            counter += 1

        loss = loss / counter

        logger.experiment.add_scalar(f"metadata/loss_model_internal_iteratoins",self.iteration,self.iteration)
        logger.experiment.add_scalar(f"parameters/epsilon_causality_weight",self.epsilon_t,self.iteration)

        return loss


class Objective_PINF_parallel_Ground_Truth_one_param_V2(Objective_PINF_parallel_Ground_Truth_one_param):
    """
    Basically the same as the default implementation, but the distance between the grid points is now considered in the computation of the causality weights.
    """

    def compute_causality_weights_exponents(self,loss:torch.tensor,param_tensor:torch.tensor)->torch.tensor:
        '''
        Compute the logarithms of the causality weights for the individual grid points.

        parameters:
            loss:               Tensor of shape [K] contaiing the loss values at the evaluated parameter values
            param_tensor:       Tensor of shape [K,self.bs] containing the parameter values at which the loss is evaluated. Has to be sorted in ascending order

        returns:
            causality_weights:  Tensor of shape [K] containing the logarithms of the causality weights for the individual losses
        '''

        assert(len(loss.shape) == 1)
        assert(len(param_tensor.shape) == 2)
        assert(loss.shape[0] == param_tensor.shape[0])

        # Compute the epsilon used for the causality weights
        self.epsilon_t = self.epsilon_causality_weight * np.exp(- self.tau_cw * (self.iteration - self.iter_start))

        with torch.no_grad():
            causality_weights = torch.zeros(param_tensor.shape[0]).to(self.device)

            for i in range(len(param_tensor)):

                param_i = param_tensor[i][0]

                # Get the index of the closest base parameter
                a = torch.argmin((self.base_params.cpu().detach() - param_i.cpu().detach()).abs()).item()
                param_base = self.base_params[a]

                idx_base = torch.where(param_tensor[:,0] == param_base)[0].item()
                idx_parameter = i

                # Get the second closest base parameter
                if len(self.base_params) > 1:
                    mask = self.base_params != param_base
                    base_params_masked = self.base_params[mask]

                    b = torch.argmin((base_params_masked.cpu().detach() - param_i.cpu().detach()).abs()).item()
                    param_base_second = base_params_masked[b]
                    idx_base_second = torch.where(param_tensor[:,0] == param_base_second)[0].item()

                # Get the loss weights based on the closest base parameter
                if idx_base < idx_parameter:
                    s1 = loss[idx_base:idx_parameter]
                    grid_distances1 = torch.abs(self.param_grid.squeeze()[idx_base+1:idx_parameter+1] - self.param_grid.squeeze()[idx_base:idx_parameter])

                elif idx_base > idx_parameter:
                    s1 = loss[idx_parameter+1:idx_base+1]
                    grid_distances1 = torch.abs(self.param_grid.squeeze()[idx_parameter:idx_base] - self.param_grid.squeeze()[idx_parameter+1:idx_base+1])

                else:
                    s1 = torch.zeros(1).to(self.device)
                    grid_distances1 = torch.zeros(1).to(self.device)

                assert(s1.shape == grid_distances1.shape)
                s1 = s1 * grid_distances1

                exponent_closest = (-s1.sum() * self.epsilon_t).detach()

                if len(self.base_params) > 1:
                    # If applicable, get the loss weights based on the second closest base parameter, i.e. if the sample is between two base parameters
                    is_between = (param_base < param_i < param_base_second) or (param_base > param_i > param_base_second)

                    if is_between and self.average_importance_weights:
                        # Get the loss weights based on the closest base parameter
                        if idx_base_second < idx_parameter:
                            s2 = loss[idx_base_second:idx_parameter]
                            grid_distances2 = torch.abs(self.param_grid.squeeze()[idx_base_second+1:idx_parameter+1] - self.param_grid.squeeze()[idx_base_second:idx_parameter])

                            d_base_idx = idx_parameter - (idx_base_second + 1)

                        elif idx_base_second > idx_parameter:
                            s2 = loss[idx_parameter+1:idx_base_second+1]
                            grid_distances2 = torch.abs(self.param_grid.squeeze()[idx_parameter:idx_base_second] - self.param_grid.squeeze()[idx_parameter+1:idx_base_second+1])

                            d_base_idx = idx_base_second - 1 - idx_parameter

                        else:
                            raise ValueError("Not suported case for causality weights")
                        
                        assert(s2.shape == grid_distances2.shape)

                        s2 = s2 * grid_distances2

                        exponent_second = (-s2.sum() * self.epsilon_t).detach()

                        # Get the relative weigting based on the distance to the base parameters
                        d_base_base = abs(idx_base_second - idx_base) - 2

                        assert(d_base_base >= 0)
                        assert(d_base_idx >= 0)
                        assert(d_base_base >= d_base_idx)

                        k = d_base_idx / d_base_base

                        #Catch edge cases where the weights ignore one of the two contributions
                        if k == 1.0:
                            exponent = exponent_closest
                        
                        elif k == 0.0:
                            exponent = exponent_second
                        
                        else:
                            exponent = torch.logsumexp(torch.tensor([exponent_closest + np.log(k),exponent_second + np.log(1-k)]),0)

                        assert(exponent.shape == torch.Size([]))

                    else:
                        exponent = exponent_closest

                else:
                    exponent = exponent_closest
                
                causality_weights[i] = exponent

            causality_weights = causality_weights.detach()

            assert(causality_weights.shape == self.multiplyer.squeeze().shape)

            causality_weights = causality_weights * self.multiplyer.squeeze().to(causality_weights.device)

            return causality_weights


class Objective_PINF_parallel_Ground_Truth_one_param_V3(Objective_PINF_parallel_Ground_Truth_one_param_V2):
    def __init__(
            self,
             param_min:float,
                 param_max:float,
                 S:Callable,
                 S_kwargs:Dict,
                 dSdparam:Callable,
                 dSdparam_kwargs:Dict,
                 n_reps:int,
                 base_params:torch.tensor,
                 device:str,
                 epsilon_causality_weight:float,
                 n_points_param_grid:int,
                 residual_processing_parameters:Dict,
                 alpha_running_EX_A:float,
                 average_importance_weights:bool,
                 update_freq_running_average_EX_A:int,
                 bs:int,
                 alpha_running_loss:float,
                 n_samples_expectation_computation:int,
                 bs_expectation_computation:int,
                 init_param_grid_in_log_space:bool,
                 n_epochs:int,
                 n_batches_per_epoch:int,
                 epsilon_causality_decay_factor:float,
                 epsilon_reduction_factor_inbetween:float,
                 use_learned_energy:bool
                 )->None:
        """
        As 'Objective_PINF_parallel_Ground_Truth_one_param_V2' but allow usage of learned energy to approximate the target energy in the case of power-scaling.

        parameters:
            param_min:                          The minimal parameter value
            param_max:                          The maximal parameter value
            S:                                  The ground truth function in the exponent of the Gibbs distribution
            S_kwargs:                           Additional parameters for the ground truth energy function
            dSdparam:                           The derivative of the ground truth function with respect to the parameter
            dSdparam_kwargs:                    Additional parameters for the derivative of the ground truth energy function
            n_reps:                             The number of repetitions for the loss computation
            base_params:                        The base parameters at which the nll loss is computed
            device:                             The device on which the computation is performed
            residual_processing_parameters:     Parametersto compute the loss from the residuals
            bs:                                 The batch size for the computation of the loss
            epsilon_causality_weight:           The weight of the causality term in the loss
            n_points_param_grid:int,            The number of points in the parameter grid
            alpha_running_EX_A:                 The running average parameter for the expectation value of the derivate of the ground truth energy function with respect to the parameter
            average_importance_weights:         If True, the importance weights are averaged if they are inbetween of two base parameters
            update_freq_running_average_EX_A:   The frequency with which the running average of the expectation value is updated
            alpha_running_loss:                 The running average parameter for the loss
            n_samples_expectation_computation:  The number of samples used to compute the expectation values
            bs_expectation_computation:         The batch size for the computation of the expectation values
            init_param_grid_in_log_space:       If True, the parameter grid is initialized in log space
            n_epochs:                           Number of epochs in which this loss is used
            n_batches_per_epoch:                Number of optimization steps per epoch
            epsilon_causality_decay_factor:     Final ration of the initial value of epsilon (reduced by applying exponential decay)
            epsilon_reduction_factor_inbetween: Factor by which epsilon is reduced in between two base parameters
            use_learned_energy:                 Set to true to allow usage of the learned energy to approximate the target distribution
        """

        print("*********************************************************************************************")
        print("Use class 'Objective_PINF_parallel_Ground_Truth_one_param_V3'")
        print("*********************************************************************************************")
        
        super().__init__(
            param_min, 
            param_max, 
            S, 
            S_kwargs, 
            dSdparam, 
            dSdparam_kwargs, 
            n_reps, 
            base_params, 
            device, 
            epsilon_causality_weight, 
            n_points_param_grid, 
            residual_processing_parameters, 
            alpha_running_EX_A, 
            average_importance_weights, 
            update_freq_running_average_EX_A, 
            bs, 
            alpha_running_loss, 
            n_samples_expectation_computation, 
            bs_expectation_computation, 
            init_param_grid_in_log_space, 
            n_epochs, 
            n_batches_per_epoch, 
            epsilon_causality_decay_factor, 
            epsilon_reduction_factor_inbetween
            )

        self.use_learned_energy = use_learned_energy

        if self.use_learned_energy:
            print("#######################################################################\n")
            print("WARNING:\n")
            print("Learned density is used for training. Only applicable for power-scaling!\n")
            print("#######################################################################\n")

            assert(len(self.base_params) == 1)

            self.S = None
            self.dSdparam = None
            self.dSdparam_kwargs = None
            self.S_kwargs = None

    def get_expectation_values(self, INN, n_samples):
        """
        Compute the expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points

        parameters:
            INN:                The INN model
            n_samples:          The number of samples used to compute the expectation values

        returns:
            EX_A_temporaray:    The approximated expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points
        """

        INN.train(False)
        
        with torch.no_grad():

            EX_A_temporaray = torch.zeros(len(self.param_grid),1).to(self.device)
            n_batches = int(self.n_samples_expectation_computation / self.bs_expectation_computation)

            for i in tqdm.tqdm(range(len(self.param_grid))):

                param = self.param_grid[i]

                log_p_x_proposal_INN = torch.zeros(self.bs_expectation_computation * n_batches).to(self.device)
                log_p_x_proposal_GT = torch.zeros(self.bs_expectation_computation* n_batches).to(self.device)
                A_proposal = torch.zeros(self.bs_expectation_computation* n_batches).to(self.device)

                for j in range(n_batches):

                    #1) Get samples from the INN
                    x_proposal_i = INN.sample(n_samples = self.bs_expectation_computation,beta_tensor = param.item())

                    #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points

                    # Use the learned energy
                    if self.use_learned_energy:
                        A_proposal_i = - 1 / self.base_params[0].item() * INN.log_prob(x_proposal_i,self.base_params[0].item()).detach()

                        log_p_x_proposal_GT_i     = param.item() / self.base_params[0].item() * INN.log_prob(x_proposal_i,self.base_params[0].item()).detach()

                    # Use the ground-truth energy
                    else:
                        A_proposal_i = self.dSdparam(x_proposal_i,**self.dSdparam_kwargs)

                        log_p_x_proposal_GT_i     = - self.S(x_proposal_i,param.item(),**self.S_kwargs)

                    assert(A_proposal_i.shape == torch.Size([self.bs_expectation_computation]))

                    #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
                    log_p_x_proposal_INN_i    = INN.log_prob(x_proposal_i,param.item())

                    assert(log_p_x_proposal_GT_i.shape == torch.Size([self.bs_expectation_computation]))
                    assert(log_p_x_proposal_INN_i.shape == torch.Size([self.bs_expectation_computation]))

                    log_p_x_proposal_GT[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_GT_i
                    log_p_x_proposal_INN[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_INN_i
                    A_proposal[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = A_proposal_i

                assert(log_p_x_proposal_INN.shape == log_p_x_proposal_GT.shape)
                assert(log_p_x_proposal_INN.shape == torch.Size([n_samples]))

                #4) compute the log likelihood ratios
                log_w = log_p_x_proposal_GT - log_p_x_proposal_INN

                assert(log_w.shape == torch.Size([n_samples]))

                #5) compute the log parition function
                log_Z = torch.logsumexp(log_w,dim = 0) - np.log(n_samples)

                assert(log_Z.shape == torch.Size([]))

                #6) Compute the importance weights
                log_omega = log_w - log_Z

                assert(log_omega.shape == A_proposal.shape)

                #7) Compute the sample based expectation value of the energy
                EX_A_temporaray[i] = (A_proposal * log_omega.exp()).mean().item()
                
            assert(EX_A_temporaray.shape == torch.Size([len(self.param_grid),1]))
    
            INN.train(True)

            return EX_A_temporaray
    
    def get_loss(self, INN, get_eval_points):
        """
        Perform the actual loss computation

        parameters:
            INN:                The INN model
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor.

        returns:
            loss:               The physics-informed loss
        """

        #############################################################################################################
        #1) Get randomly select a batch of parameters for the evaluation
        #############################################################################################################

        # Initial call: Uniformly sample the points
        if self.log_causality_weights is None:
            print("Initial call")
            idx = torch.randint(low = 0,high = len(self.param_grid),size = (self.bs,))

        # Follow up calls: Sample the indices based on the distribution defined by the causality weights 
        else:
            m = torch.distributions.categorical.Categorical(logits = self.log_causality_weights)
            idx = m.sample([self.bs]).cpu()
        
        assert(idx.shape == torch.Size([self.bs]))

        param_tensor = self.param_grid[idx].to(self.device)

        assert(param_tensor.shape == torch.Size([self.bs,1]))

        #############################################################################################################
        #2) Compute the target for the logarithm of the INN distribution
        #############################################################################################################
        with torch.no_grad():

            INN.train(False)

            #2a) Get evaluation points at which the loss is evaluated
            x_eval = get_eval_points(beta_tensor = param_tensor)

            #2b) Compute the ground truth energies of the evaluation points

            # Use learned energy
            if self.use_learned_energy:
                A_eval = - 1 / self.base_params[0].item() * INN.log_prob(x_eval,self.base_params[0].item()).detach().reshape(-1,1)
            
            # Use ground truth energy
            else:
                A_eval = self.dSdparam(x_eval,**self.dSdparam_kwargs).reshape(-1,1)

            assert(A_eval.shape == param_tensor.shape)

            #2c) Compute the target
            target = (self.EX_A[idx] - A_eval).detach()
        
            assert(self.EX_A[idx].shape == torch.Size([self.bs,1]))
            assert(target.shape == torch.Size([self.bs,1]))

            INN.train(True)

        #############################################################################################################
        #3) Compute the gradient of the INN ditsribution with respect to the parameter
        #############################################################################################################
        param_tensor.requires_grad_(True)

        #3a) Compute the log likelihood of the evaluation points under the INN distribution
        log_p_x_eval = INN.log_prob(x_eval,param_tensor)

        #3b) Compute the gradient of the log likelihood of the evaluation points under the INN distribution with respect to the parameter
        grad = torch.autograd.grad(log_p_x_eval.sum(),param_tensor,create_graph=True)[0]

        assert(grad.shape == param_tensor.shape)
        assert(grad.shape == target.shape)

        #############################################################################################################
        #4)Compute loss for each of the evaluation points
        #############################################################################################################
        
        #4a) Get the residuals
        residuals = grad - target.detach()

        assert(residuals.shape == param_tensor.shape)

        #4b) Compute the loss based on the residuals
        loss = get_loss_from_residuals(residuals,dim = 1,**self.residual_processing_parameters)

        assert(loss.shape == torch.Size([param_tensor.shape[0]]))

        #############################################################################################################
        #5) Update the running averages of the loss on the evaluated grid points
        #############################################################################################################

        #5a) Average the losses if there are multiple samples for the same parameter
        unique_idx,position_uniques,counts_uniques = torch.unique(idx,return_counts = True,return_inverse = True)

        target_tensor_loss = torch.zeros(unique_idx.shape[0])
        target_tensor_loss.scatter_add_(0, position_uniques, loss.cpu().detach()) / counts_uniques
        target_tensor_loss = target_tensor_loss / counts_uniques

        #5b) Compute weights depending on the number of samples for the same parameter
        alphas_reweighted = self.alpha_running_loss**counts_uniques  

        #5c) Compute running average
        assert(self.loss_statistics[unique_idx].shape == target_tensor_loss.shape)
        assert(self.loss_statistics[unique_idx].shape == alphas_reweighted.shape)

        w_old = alphas_reweighted
        w_new = 1.0 - w_old

        assert(w_old.shape == alphas_reweighted.shape)
        assert(w_new.shape == w_old.shape)

        self.loss_statistics[unique_idx.cpu()] = w_old * self.loss_statistics[unique_idx] + w_new * target_tensor_loss.cpu()

        assert(self.loss_statistics.shape == torch.Size([len(self.param_grid)]))

        #############################################################################################################
        #6) If applicable, update the logarithms of the causality weights
        #############################################################################################################
        if (self.log_causality_weights is None) or ((self.iteration % self.freq_update_causality_weights) == 0):
            self.log_causality_weights = self.compute_causality_weights_exponents(self.loss_statistics.detach(),self.param_grid)

        #############################################################################################################
        #7) Aggregate the loss
        #############################################################################################################
        loss = loss.mean()
        
        assert(loss.shape == torch.Size([]))

        return loss
    