import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from pinf.models.INN import INN_Model

from pinf.models.utils import (
    condition_specs_dict,
    activation_dict
)

from pinf.models.subnetworks import (
    WrappedConditionalCouplingBlock,
    MLPEmbedder,
    ConditionalScalingLayer,
    constructor_subnet_fc_configured,
    constructor_conditional_scaling_subnet_fc_configured,
    constructor_subnet_2D_conv_configured
)

def set_up_sequence_INN_ScalarTheory(config:dict,training_set:torch.utils.data.Dataset = None)->INN_Model:
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

    inn = Ff.SequenceINN(1,config["config_data"]["N"],config["config_data"]["N"])

    # Modify the configuration file for the condition in case of learnable embedding
    embedding_model = None
    if config["config_model"]["process_beta_parameters"]["mode"] == "learnable":

        condition_specs_dict["learnable"] = [0,(config["config_model"]["process_beta_parameters"]["learnable_temperature_embedding_dim"],)]

        # Initialize the embedding model for the inverse temperature
        embedding_model = MLPEmbedder(
            d_hidden = config["config_model"]["process_beta_parameters"]["d_hidden"],
            d_out = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][1][0],
            activation = activation_dict[config["config_model"]["process_beta_parameters"]["activation_function_type"]]
            )
    
        embedding_model.to(config["device"])

    # Get the shape of the condition passed to the coupling blocks
    cond_num = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][0]
    cond_shape = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][1]

    output_str += "\tBeta Processing:\t\t" + config["config_model"]["process_beta_parameters"]["mode"] + "\n"
    output_str += f"\tCondition specs:\t\tidx:{cond_num}\tshape:{cond_shape}\n"

    ######################################################################################################################################
    # Add learnable global scaling to the network
    ######################################################################################################################################

    inn.append(
        ConditionalScalingLayer,
        subnet_constructor = constructor_conditional_scaling_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc_learnable_scaling"],activation_type = config["config_model"]["activation_function_type"]),
        cond_shape = cond_shape,
        cond = cond_num
        )
        
    ######################################################################################################################################
    #Add the coupling blocks with convolutional subnetworks
    ######################################################################################################################################
    for i in range(config["config_model"]["n_stages_conv"]):

        # Add downsampling and increase the number of channels accordingly
        inn.append(Fm.IRevNetDownsampling)

        # Get the number of data channels in this stage
        n_channels_hidden_i = config["config_model"]["n_channels_conv_hidden_list"][i]

        # Add one RQ spline block
        inn.append(
            WrappedConditionalCouplingBlock,
            BlockType = Fm.RationalQuadraticSpline,
            subnet_constructor = constructor_subnet_2D_conv_configured(c_hidden = n_channels_hidden_i,activation_type = config["config_model"]["activation_function_type"]),
            cond_shape = cond_shape,
            cond = cond_num,
            )
        inn.append(Fm.PermuteRandom)

        # Add affine coupling blocks
        for j in range(config["config_model"]["affine_coupling_block_number_per_stage_conv"]):
            
            if config["config_model"]["use_act_norm"]:
                inn.append(Fm.ActNorm)

            # Concatenete the condition to the input of the coupling block as additional channels, one per condition dimension
            inn.append(
                WrappedConditionalCouplingBlock,
                BlockType = Fm.AllInOneBlock,
                subnet_constructor = constructor_subnet_2D_conv_configured(c_hidden = n_channels_hidden_i,activation_type = config["config_model"]["activation_function_type"]),
                cond_shape = cond_shape,
                cond = cond_num,
                permute_soft = True,
                )

    ######################################################################################################################################
    # Add the coupling blocks with fully connected subnetworks
    ######################################################################################################################################
    inn.append(Fm.Flatten)

    inn.append(
        WrappedConditionalCouplingBlock,
        BlockType = Fm.RationalQuadraticSpline,
        subnet_constructor = constructor_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc"],activation_type = config["config_model"]["activation_function_type"]),
        cond_shape = cond_shape,
        cond = cond_num,
        )

    inn.append(Fm.PermuteRandom)

    for j in range(config["config_model"]["affine_coupling_block_number_per_stage_fc"]):

        if config["config_model"]["use_act_norm"]:
            inn.append(Fm.ActNorm)

        inn.append(
            WrappedConditionalCouplingBlock,
            BlockType = Fm.AllInOneBlock,
            subnet_constructor = constructor_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc"],activation_type = config["config_model"]["activation_function_type"]),
            cond_shape = cond_shape,
            cond = cond_num,
            permute_soft = True
            )

    # Set the model to the device
    inn.to(config["device"])

    ######################################################################################################################################
    # Wrapper for all INN operations
    ######################################################################################################################################
    INN = INN_Model(
        d = config["config_data"]["N"] * config["config_data"]["N"], 
        inn = inn, 
        device = config["device"],
        latent_mode=config["config_model"]["latent_mode"],
        process_beta_mode = config["config_model"]["process_beta_parameters"]["mode"],
        embedding_model=embedding_model
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
