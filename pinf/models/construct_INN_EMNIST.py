import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch

from pinf.models.INN import INN_Model
from pinf.models.utils import (
    coupling_block_dict,
)
from pinf.models.subnetworks import (
    WrappedConditionalCouplingBlock,
    construct_MNIST_Conv_subnet_configured,
    construct_MNIST_FC_subnet_configured
)

def set_up_sequence_INN_MNIST_like(config:dict,training_set:torch.utils.data.Dataset = None)->INN_Model:
    """
    Initialisation of an INN object based on a FrEIA sequence INN.

    parameters:
        config:         Conifguration file to set up the model
        training_set:   Training set

    return:
        INN:            Initialized wrapper for INN operations
    """

    #Initialize the inveritble function
    inn = Ff.SequenceINN(1,28,28)

    #Get the coupling block
    coupling_block_class = coupling_block_dict[config["config_model"]["coupling_block_type"]]

    #Get the specifications for the conditioning of the model
    if config["config_model"]["conditional"]:
        cond_num = 0
        cond_shape = (config["config_data"]["n_classes"],)
        process_beta_mode = "one_hot"
    else:
        cond_num = None
        cond_shape = None
        process_beta_mode = "ignore_beta"

    #Add coupling blocks to the invertible function
    for i in range(config["config_model"]["n_stages_conv"]):

        #Add downsampling and increase the number of channels accordingly
        inn.append(Fm.IRevNetDownsampling)

        #Get the number of data channels in this stage
        n_channels_hidden_i = config["config_model"]["n_channels_conv_hidden_list"][i]

        for j in range(config["config_model"]["coupling_block_number_per_stage_conv"]):

            #Add act norm
            if config["config_model"]["use_act_norm"]:
                inn.append(Fm.ActNorm)

            inn.append(
                WrappedConditionalCouplingBlock,
                BlockType = coupling_block_class,
                subnet_constructor = construct_MNIST_Conv_subnet_configured(c_hidden = n_channels_hidden_i,activation_type = config["config_model"]["activation_function_type"]),
                cond_shape = cond_shape,
                cond = cond_num,
                **config["config_model"]["coupling_block_params"]
                )
            
            #Add permutation
            if config["config_model"]["coupling_block_type"] != "AllInOne":
                inn.append(Fm.PermuteRandom)

    #Add flattening
    inn.append(Fm.Flatten)

    #Add coupling blocks with fully connected subnetworks
    for j in range(config["config_model"]["coupling_block_number_per_stage_fc"]):
        inn.append(
            WrappedConditionalCouplingBlock,
            BlockType = coupling_block_class,
            subnet_constructor = construct_MNIST_FC_subnet_configured(c_hidden = config["config_model"]["d_hidden_fc"],activation_type = config["config_model"]["activation_function_type"]),
            cond_shape = cond_shape,
            cond = cond_num,
            **config["config_model"]["coupling_block_params"]
            )

        #Add permutation
        if config["config_model"]["coupling_block_type"] != "AllInOne":
            inn.append(Fm.PermuteRandom)
    
    #Set the invertible function to the device
    inn.to(config["device"])

    #Initialize the INN model
    INN = INN_Model(
        d = 28*28,
        inn = inn,
        device=config["device"],
        latent_mode="standard_normal",
        process_beta_mode = process_beta_mode
    )

    return INN
