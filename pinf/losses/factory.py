from pinf.datasets.gradients import (
    dS_dparam_dict_multiple_parameters,
    dS_dparam_dict
)

from pinf.datasets.energies import (
    S_dict_multiple_parameters,
    S_dict
)

from pinf.datasets.log_likelihoods import (
    log_p_target_dict_multiple_parameters,
    log_p_target_dict
    )

from pinf.losses.reverse_KL import (
    Objective_reverse_KL,
    Objective_reverse_KL_Multiple_Parameters
    )

from pinf.losses.TRADE_grid_less import (
    Objective_PINF_local_Ground_Truth_one_param_V2,
    Objective_PINF_local_Ground_Truth_one_param_V3,
    Objective_PINF_local_Ground_Truth_Multiple_Parameters
    )

from pinf.losses.TRADE_grid_based import (
    Objective_PINF_parallel_Ground_Truth_one_param_V2,
    Objective_PINF_parallel_Ground_Truth_one_param_V3
)

#####################################################################
# Multiple external conditions
#####################################################################

class DataFreeLossFactory_MultipleParameters():
    def __init__(self):
        pass

    def create(self,key,config):

        # Check consistency
        assert(key == config["config_training"]["data_free_loss_mode"])

        #################################################################################
        # Reverse KL training
        #################################################################################

        if key == "reverse_KL_multi_param":
            
            loss_model = Objective_reverse_KL_Multiple_Parameters(
                log_p_target=log_p_target_dict_multiple_parameters[config["config_training"]["log_p_target_name"]],
                log_p_target_kwargs=config["config_training"]["log_p_target_kwargs"],
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )

        #################################################################################
        # TRADE without discretized condition
        #################################################################################

        elif key == "TRADE_no_grid_multi_param":

            dSdparam_list = []

            for dS_dparam_name in config["config_training"]["dS_dparam_names"]:
                dSdparam_list.append(dS_dparam_dict_multiple_parameters[dS_dparam_name])

            loss_model = Objective_PINF_local_Ground_Truth_Multiple_Parameters(
                S =  S_dict_multiple_parameters[config["config_data"]["data_set_name"]],
                dSdparam_list = dSdparam_list,
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )
                        
        else:
            loss_model = None
            print("No data-free loss contribution is used")

        return loss_model

#####################################################################
# One external condition
#####################################################################

class DataFreeLossFactory():
    def __init__(self):
        pass

    def create(self,key,config):

        #Check consistency
        assert(key == config["config_training"]["data_free_loss_mode"])

        #################################################################################
        # Reverse KL training
        #################################################################################

        if key == "reverse_KL":
            loss_model = Objective_reverse_KL(
                log_p_target=log_p_target_dict[config["config_training"]["log_p_target_name"]],
                log_p_target_kwargs=config["config_training"]["log_p_target_kwargs"],
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )
            print("Initialize loss model of type <Objective_reverse_KL>")

        #################################################################################
        # TRADE without discretized condition
        #################################################################################

        elif key == "PINF_local_Ground_Truth_one_param_V2":
            loss_model = Objective_PINF_local_Ground_Truth_one_param_V2(
                S = S_dict[config["config_data"]["data_set_name"]],
                dSdparam = dS_dparam_dict[config["config_data"]["data_set_name"]],
                device = config["device"],
                **config["config_training"]["loss_model_params"]               
            )
            print("Initialize loss model of type <Objective_PINF_local_Ground_Truth_one_param_V2>")

        elif key == "PINF_local_Ground_Truth_one_param_V3":
            loss_model = Objective_PINF_local_Ground_Truth_one_param_V3(
                S = S_dict[config["config_data"]["data_set_name"]],
                dSdparam = dS_dparam_dict[config["config_data"]["data_set_name"]],
                device = config["device"],
                **config["config_training"]["loss_model_params"]               
            )
            print("Initialize loss model of type <Objective_PINF_local_Ground_Truth_one_param_V3>")
        
        #################################################################################
        # TRADE with discretized condition
        #################################################################################

        elif key == "PINF_parallel_Ground_Truth_one_param_V2":

            loss_model = Objective_PINF_parallel_Ground_Truth_one_param_V2(
                S = S_dict[config["config_data"]["data_set_name"]],
                dSdparam = dS_dparam_dict[config["config_data"]["data_set_name"]],
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )

            print("Initialize loss model of type <Objective_PINF_parallel_Ground_Truth_one_param_V2>")

        elif key == "PINF_parallel_Ground_Truth_one_param_V3":

            loss_model = Objective_PINF_parallel_Ground_Truth_one_param_V3(
                S = S_dict[config["config_data"]["data_set_name"]],
                dSdparam = dS_dparam_dict[config["config_data"]["data_set_name"]],
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )

            print("Initialize loss model of type <Objective_PINF_parallel_Ground_Truth_one_param_V3>")

        else: 
            loss_model = None
            print("No data-free loss contribution is used")

        return loss_model