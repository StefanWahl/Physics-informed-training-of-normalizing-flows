import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from pinf.models.INN import INN_Model

from pinf.models.utils import (
    condition_specs_dict,
    coupling_block_dict,
)

from pinf.models.subnetworks import (
    WrappedConditionalCouplingBlock,
    constructor_subnet_fc_configured
)

def set_up_sequence_INN_2D_GMM(config:dict,training_set:torch.utils.data.Dataset = None)->INN_Model:
    """
    Initialisation of an INN object based on a FrEIA sequence INN.

    parameters:
        config:         Conifguration file to set up the model
        training_set:   Training set

    return:
        INN:            Initialized wrapper for INN operations
    """

    # sCollect information for logging
    output_str = "*********************************************************************************************\n"
    output_str += "\nINFO:\n\nInitialize sequence INN\n\n"
    output_str += f"\tModel on device \t{config['device']}\n"

    ######################################################################################################################################
    #Initialize the invertible function
    ######################################################################################################################################

    inn = Ff.SequenceINN(config["config_data"]["init_data_set_params"]["d"])

    # Get the coupling block class used internally
    coupling_block_class = coupling_block_dict[config["config_model"]["coupling_block_type"]]

    # Get the shape of the condition passed to the coupling blocks
    cond_num = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][0]
    cond_shape = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][1]

    output_str += "\tBeta Processing:\t\t" + config["config_model"]["process_beta_parameters"]["mode"] + "\n"
    output_str += f"\tCondition specs:\t\tidx:{cond_num}\tshape:{cond_shape}\n"

    ######################################################################################################################################
    # Add the coupling blocks with fully connected subnetworks
    ######################################################################################################################################

    for j in range(config["config_model"]["coupling_block_number_per_stage_fc"]):

        # Add ActNorm
        if config["config_model"]["use_act_norm"]:
            inn.append(Fm.ActNorm)

        # Concatenete the condition to the input of the coupling block as additional channels, one per condition dimension
        if config["config_model"]["condition_inclusion_fc_layers"] == "concatenate":

            inn.append(
                WrappedConditionalCouplingBlock,
                BlockType = coupling_block_class,
                subnet_constructor = constructor_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc"],activation_type = config["config_model"]["activation_function_type"]),
                cond_shape = cond_shape,
                cond = cond_num,
                **config["config_model"]["coupling_block_params"]
                )

        else:
            raise NotImplementedError()

        # Add permutation
        if config["config_model"]["coupling_block_type"] != "AllInOne":
            inn.append(Fm.PermuteRandom)

    # Send the model to the device
    inn.to(config["device"])

    ######################################################################################################################################
    # Wrapper for all INN operations
    ######################################################################################################################################

    INN = INN_Model(
        d = config["config_data"]["init_data_set_params"]["d"], 
        inn = inn, 
        device = config["device"],
        latent_mode=config["config_model"]["latent_mode"],
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
