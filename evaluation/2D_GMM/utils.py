import yaml
import os
from pathlib import Path
from torch.utils.data import DataLoader

from pinf.models.construct_INN_2D_GMM import set_up_sequence_INN_2D_GMM
from pinf.datasets.datasets import DataSet2DGMM

def p_beta(x,beta,gmm,Z = None):
    q_beta = gmm(x).pow(beta)

    if Z is None:
        return q_beta
    
    else:
        return q_beta / Z

def load_INN(base_path:str,use_last:bool = False,device:str = "cuda:0"):

    config_i = yaml.safe_load(Path(base_path + "/hparams.yaml").read_text())
    state_dict_folder_i = base_path + f"/checkpoints/"

    files = os.listdir(state_dict_folder_i)

    
    for f in files:
        print(f)
        #Use the last recorded state dict
        if use_last:

            if f == "last.ckpt":
                state_dict_path_i = os.path.join(state_dict_folder_i,f)
                break

        #Use the best performing state dict
        else:
            if f.startswith("checkpoint_epoch"):
                state_dict_path_i = os.path.join(state_dict_folder_i,f)
                break

    config_i["device"] = device

    INN_i = set_up_sequence_INN_2D_GMM(config=config_i)
    INN_i.load_state_dict(state_dict_path_i)
    INN_i.train(False)

    print(state_dict_path_i)

    return INN_i,config_i

def get_validation_loader_dict_2D_GMM(T_list_eval,n_samples):
    validation_data_loader_dict = {}

    for i,T_i in enumerate(T_list_eval):

        T_i = round(T_i,5)
        print(f"Loading validation data for T = {T_i}")

        DS_i = DataSet2DGMM(
            d = 2,
            mode = "validation",
            temperature_list=[T_i],
            base_path="../../data/2D_GMM/",
            n_samples=n_samples
            )

        DL_i = DataLoader(
            DS_i,
            batch_size = 1000,
            shuffle = False,
            num_workers = 4
        )

        validation_data_loader_dict[f"{T_i}"] = DL_i

    return validation_data_loader_dict
