
from typing import Tuple
import torch
import numpy as np

def get_beta(t:int,t_burn_in:int,t_full:int,beta_star:float,beta_min:float,beta_max:float,**kwargs)->Tuple[float,float,float]:
        """
        Sample the inverse temperature from an interval depending on the current time step

        parameters:
            t:              The current time step
            t_burn_in:      The lenght of the burin in phase
            t_full:         The number of time steps it takes to reach the full range
            beta_star:      The base condition value
            beta_min:       The minimal condition value
            beta_max:       The maximal condition value

        returns:
            beta:           The sampled condition value
            left:           The lower boundary of the interval from which the inverse temperature is sampled
            right:          The upper boundary of the interval from which the inverse temperature is sampled
        """

        assert(isinstance(beta_star,float))
        assert(isinstance(beta_min,float))
        assert(isinstance(beta_max,float))
        assert(t_burn_in <= t_full)
        assert(beta_star >= beta_min)
        assert(beta_star <= beta_max)
        assert(beta_star > 0.0)
        assert(beta_max > 0.0)
        assert(beta_min > 0.0)

        #Burn in Phase return the inverse base temperature
        if t < t_burn_in:
            beta = beta_star
            left = beta_star
            right = beta_star

        #Randomly sample the inverse temperature from the interval
        else:
            
            #Compute the linear interpolation factor
            #use full range from the begining or directly after the burn in phase
            if (t_full == 0) or (t_full == t_burn_in): l = 1.0
            
            #There is a finite ramp up phase
            else:
                l = (min(t,t_full)-t_burn_in) / (t_full-t_burn_in)

            #Sample the inverse temperature uniformly from the interval [beta_star - l * beta_min, beta_star + l * beta_max]
            if kwargs["mode"] == "linear":

                left = beta_star - (beta_star -beta_min) * l
                right = beta_star + (beta_max -beta_star) * l

                beta = (right - left) * torch.rand(1).item() + left

            #Sample the logarithm of the inverse temperature uniformly from the interval [log(beta_star) - l * log(beta_min), log(beta_star) + l * log(beta_max)]
            elif kwargs["mode"] == "log-linear":

                beta_min_log = np.log(beta_min)
                beta_max_log = np.log(beta_max)
                beta_0_log = np.log(beta_star)

                log_left = beta_0_log - (beta_0_log - beta_min_log) * l
                log_right = beta_0_log + (beta_max_log - beta_0_log) * l

                log_beta_t = (log_right - log_left) * torch.rand(1).item() + log_left

                beta = np.exp(log_beta_t)
                left = np.exp(log_left)
                right = np.exp(log_right)

            else:
                raise ValueError("Sampling mode for inverse temperature not recognized")
            
        return beta,left,right

def get_loss_from_residuals(residuals:torch.Tensor,dim:int = 0,**kwargs) -> torch.Tensor:
    """
    Compute the loss from the residuals

    parameters:
        residuals:  One dimensional tensor containing the residuals
        dim:        Dimension along which the residuals are averaged

    returns:
        loss:       The loss
    """

    #Mean squared error
    if kwargs["mode"] == "MSE":
        loss = residuals.pow(2)

    #Mean absolute error
    elif kwargs["mode"] == "MAE":
        loss = residuals.abs()

    #Huber loss
    elif kwargs["mode"] == "Huber":
        loss = huber_loss(residuals,delta = kwargs["delta"])
    
    else:
        raise ValueError("Loss computation mode not recognized")
    

    return loss.mean(dim = dim)

def huber_loss(x:torch.Tensor,delta:float) -> torch.Tensor:
    """
    Compute the Huber loss

    parameters:
        x       The residuals
        delta   The threshold for the Huber loss

    returns:
        h: The Huber loss
    """

    assert(isinstance(delta,float))

    h = torch.where(x.abs() < delta,0.5 * x.pow(2),delta * (x.abs() - 0.5 * delta))
    
    return h

