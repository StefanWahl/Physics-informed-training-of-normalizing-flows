import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from pinf.models.INN import INN_Model__MultipleExternalParameters

from pinf.models.utils import (
    coupling_block_dict,
)

from pinf.models.subnetworks import (
    WrappedConditionalCouplingBlock,
    FixedGlobalScaling,
    constructor_subnet_fc_configured
)

def set_up_sequence_INN_2D_ToyExample_two_parameters(config:dict,training_set:torch.utils.data.Dataset = None)->INN_Model__MultipleExternalParameters:
    """
    Initialisation of an INN object based on a FrEIA sequence INN.

    parameters:
        config:         Conifguration file to set up the model
        training_set:   Training set

    return:
        INN:            Initialized wrapper for INN operations
    """

    # Collect information for logging
    output_str = "*********************************************************************************************\n"
    output_str += "\nINFO:\n\nInitialize sequence INN\n\n"
    output_str += f"\tModel on device \t{config['device']}\n"

    ######################################################################################################################################
    # Initialize the invertible function
    ######################################################################################################################################

    inn = Ff.SequenceINN(config["config_data"]["init_data_set_params"]["d"])

    std = torch.std(training_set.data.float())
    mean = torch.mean(training_set.data.float())
    inn.append(FixedGlobalScaling,alpha = 1 / std,beta = - mean / std)

    # Get the coupling block class used internally
    coupling_block_class = coupling_block_dict[config["config_model"]["coupling_block_type"]]

    # Get the shape of the condition passed to the coupling blocks
    if config["config_model"]["process_beta_parameters"]["mode"] != "ignore_beta":
        cond_num = 0
        cond_shape = (config["config_model"]["process_beta_parameters"]["output_dim_condition_trafo"],)

    else:
        cond_num = None
        cond_shape = None

    output_str += "\tBeta Processing:\t\t" + config["config_model"]["process_beta_parameters"]["mode"] + "\n"
    output_str += f"\tCondition specs:\t\tidx:{cond_num}\tshape:{cond_shape}\n"

    ######################################################################################################################################
    # Add the coupling blocks with fully connected subnetworks
    ######################################################################################################################################

    for j in range(config["config_model"]["coupling_block_number_per_stage_fc"]):

        # Add act norm
        if config["config_model"]["use_act_norm"]:
            inn.append(Fm.ActNorm)

        # Concatenete the condition to the input of the coupling block as additional channels, one per condition dimension
        inn.append(
            WrappedConditionalCouplingBlock,
            BlockType = coupling_block_class,
            subnet_constructor = constructor_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc"],activation_type = config["config_model"]["activation_function_type"]),
            cond_shape = cond_shape,
            cond = cond_num,
            **config["config_model"]["coupling_block_params"]
            )
    
        # Add permutation
        if config["config_model"]["coupling_block_type"] != "AllInOne":
            inn.append(Fm.PermuteRandom)

    # Send the model to the device
    inn.to(config["device"])

    ######################################################################################################################################
    # Wrapper for all INN operations
    ######################################################################################################################################
    INN = INN_Model__MultipleExternalParameters(
        d = config["config_data"]["init_data_set_params"]["d"], 
        inn = inn, 
        device = config["device"],
        process_beta_mode = config["config_model"]["process_beta_parameters"]["mode"]
        )

    ######################################################################################################################################
    # Summary
    ######################################################################################################################################

    num_params = sum(p.numel() for p in INN.inn.parameters())
    output_str += f"\nNumber of model parameters: {num_params}"

    # Print information
    output_str += "\n*********************************************************************************************\n"
    if config["verbose"]: print(output_str)

    return INN
