import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
from typing import Union,Callable
from functools import partial
import torch.nn as nn
import numpy as np

from pinf.models.utils import activation_dict

######################################################################################################################################
# Invertible blocks for INN
######################################################################################################################################

class WrappedConditionalCouplingBlock(Fm.InvertibleModule):
    def __init__(self,dims_in,BlockType,subnet_constructor,dims_c = None,**kwargs):
        """
        Wrap a predefined coupling block and rehsape the condition to the required shape.

        parameters:
            dims_in:            Tupel containing a list representing the shape of an input instance.
            BlockType:          Predefined coupling block
            subnet_constructor: Sonstructor for the subnetd of the coupling blocks
            dims_c:             Dimensinality of the condition as passed to this block during a call. Not the dimensinality of the condition expected by the internal block.
        """
        super().__init__(dims_in,dims_c)

        self.dims_c = dims_c
        self.dims_in = dims_in

        # Unconditional block
        if self.dims_c is None:
            self.dims_c_internal = []

        # Conditional block
        else:
            # Shape of the condition as expected by the internal block
            self.dims_c_internal = [(dims_c[0][0],*dims_in[0][1:])]

        # Initialize the coupling block
        self.block = BlockType(
            dims_in = dims_in,
            dims_c = self.dims_c_internal,
            subnet_constructor = subnet_constructor,
            **kwargs)
        
    def transform_condition(self,c:torch.tensor)->torch.tensor:
        """
        Reshape the condition of shape [N,dims_c] in to a condition of shape [N,dims_c,H,W]

        parameters:
            c:              Condition of shape [N,dims_C] 

        returns:
            c_recomputed:   Condition tensor of shape [N,dims_C,H,W] where the i th channel contains only the number at 
                            the i th position of the input. 
        """

        # Create conditional channels in case of convolutional subnetworks
        # Convolutional subnet
        if len(self.dims_in[0]) > 1: 
            c_recomputed = c.reshape(c.shape[0],c.shape[1],1,1).expand(-1,-1,self.dims_in[0][1],self.dims_in[0][2])
        
        # No action needed in case of fully connected subnetworks
        else:
            c_recomputed = c

        return c_recomputed

    def forward(self,x_or_z, c:Union[list[torch.tensor],None] = None, rev:bool = False, jac:bool = True)->tuple:
        """
        Model call

        parameters:
            x_or_z:     Data batch
            c:          Condition
            rev:        Invert function
            jac:        Return log Jacobian determinant

        return:
            output:     Output of the internally used coupling block
        """

        # Reshape the condition passed to the block
        if self.dims_c is not None:
            reomputed_c = [self.transform_condition(c = c[0])]

        else:
            reomputed_c = None

        # Pass it through the internal coupling block
        output = self.block(
            x_or_z, 
            c = reomputed_c, 
            rev = rev, 
            jac = jac
        )
        
        return output

    def output_dims(self,input_dims):
        """
        Compute the dimensionality of the output of the block
        """
        return input_dims

class FixedGlobalScaling(Fm.InvertibleModule):
    def __init__(self,dims_in,alpha = None,beta = None,dims_c = None):
        """
        Global scaling and offset 

        parameters:
            dims_in:            Tupel containing a list representing the shape of an input instance.
            alpha:              Global scaling
            beta:               Global offset
            dims_c:             Dimensinality of the condition as passed to this block during a call. Not the dimensinality of the condition expected by the internal block.
        """
        super().__init__(dims_in,dims_c)

        self.alpha = alpha
        self.beta = beta

    def forward(self,x_or_z,c = None,rev = False,jac = True):

        # Used for sampling
        if rev:
            y = (x_or_z[0] - self.beta) / self.alpha
            jac = - np.log(self.alpha) * torch.prod(torch.tensor(x_or_z[0].shape[1:]))
        
        # Use for density evaluation
        else:
            y =  x_or_z[0] * self.alpha + self.beta
            jac = np.log(self.alpha) * torch.prod(torch.tensor(x_or_z[0].shape[1:]))

        assert(y.shape == x_or_z[0].shape)

        return (y,),jac * torch.ones(x_or_z[0].shape[0]).to(x_or_z[0].device)
    
    def output_dims(self,input_dims):
        return input_dims

######################################################################################################################################
# Constructors for subnetworks
######################################################################################################################################

def constructor_subnet_fc_plain(c_in:int,c_out:int,c_hidden:int,activation_type:str)->nn.Module:
    """
    Initialize a fully connected neural network.

    parameters:
        c_in:               Number of input channels
        c_out:              Number of output channels
        c_hidden:           Dimensionality of the hidden layers
        activation_type:    String specifying the non-linear function 
    
    return:
        layers:             Fully connected neural network
    """

    # Get the non-linearity
    activation = activation_dict[activation_type]

    layers = nn.Sequential(
        nn.Linear(c_in, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_out)
        )
    
    # Initialize the weights of the linear layers
    for layer in layers:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    # Set the weights and the bias of the final convolution to zero
    layers[-1].weight.data.fill_(0.0)
    layers[-1].bias.data.fill_(0.0)

    return layers

def constructor_subnet_fc_configured(d_hidden:int,activation_type:str)->Callable:
    return partial(constructor_subnet_fc_plain,c_hidden = d_hidden,activation_type = activation_type)
