import argparse
import json
from datetime import date
import torch
from torch.utils.data import DataLoader
import lightning as L
import os
import shutil
import numpy as np
import random
import yaml
from pathlib import Path
import tqdm

from pinf.models.construct_INN_ScalarTheory import set_up_sequence_INN_ScalarTheory
from pinf.datasets.datasets import DataSetScalarTheory2D_preprocessed_data
from pinf.datasets.energies import S_ScalarTheory
from pinf.plot.utils import (
    bootstrap,
    get_susceptibility,
    get_U_L
)

INN_constructor_dict = {
    "set_up_sequence_INN_ScalarTheory":set_up_sequence_INN_ScalarTheory
}

def get_ESS_r(log_p_theta_INN:torch.Tensor,log_p_target_INN:torch.Tensor)->float:
    """
    Compute teh relative effective sample size.

    parameters:
        log_p_theta_INN:    Log-likelihoods under the model distribution of model samples
        log_p_target_INN    Log-likelihoods under the target distribution of model samples

    returns:
        ESS_r               Approximated relative effective sample size.
    """

    # Compuete the relative Kish effective sample size
    log_omega = log_p_target_INN - log_p_theta_INN
    log_a = 2 * torch.logsumexp(log_omega,0)
    log_b = torch.logsumexp(2 * log_omega,0)

    ESS_r = (torch.exp(log_a - log_b) / len(log_omega)).item()

    return ESS_r

def load_model_from_folder(folder:str,device:str,use_last:bool)->tuple:
    """
    Load a pretrained normalizing flow.

    parameters:
        folder:     Location of the experiment
        device:     Device on which the experiment runs.
        use_last:   Load the model observed at the end of the training if set to True.

    returns:
        (INN_i,config_i):   Initialized INN model and parameter dictionary.
    """

    config_i = yaml.safe_load(Path(folder + "/hparams.yaml").read_text())
    state_dict_folder_i = folder + f"/checkpoints/"

    files = os.listdir(state_dict_folder_i)

    for f in files:

        # Use the last recorded state dict
        if use_last:

            if f == "last.ckpt":
                state_dict_path_i = os.path.join(state_dict_folder_i,f)
                break

        # Use the best performing state dict
        else:
            if f.startswith("checkpoint_epoch"):
                state_dict_path_i = os.path.join(state_dict_folder_i,f)
                break

    config_i["device"] = device

    INN_i = INN_constructor_dict[config_i["config_model"]["set_up_function_name"]](config=config_i)
    INN_i.load_state_dict(state_dict_path_i)
    INN_i.train(False)

    print(state_dict_path_i)

    return INN_i,config_i

def run_evaluation(experiment_folder:str,args_dict:dict)->None:
    """
    Run the full evaluation of the trained model.

    parameters:
        experiment_folder:  Location of the experiment.
        args:               Arguments for the evaluation.
    """

    # Folder to store the results
    if args_dict["use_last"]:
        results_folder = os.path.join(experiment_folder,"Evaluation_last")

    else:
        results_folder = os.path.join(experiment_folder,"Evaluation_best")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    else:
        pass

    # Store the metadata of the evaluation
    with open(os.path.join(results_folder,"evaluation_meta_data.json"),"w") as f:
        json.dump(args_dict,f)
    f.close()

    device = "cuda:0"

    # Load the best performing mode
    INN,config = load_model_from_folder(
        folder = experiment_folder,
        device=device,
        use_last=args_dict["use_last"]
    )

    # Get validation data
    kappa_val_data = np.arange(config["config_evaluation"]["kappa_validation_specs"][0],config["config_evaluation"]["kappa_validation_specs"][1],config["config_evaluation"]["kappa_validation_specs"][2])
    lambda_val_data = np.arange(config["config_evaluation"]["lambda_validation_specs"][0],config["config_evaluation"]["lambda_validation_specs"][1],config["config_evaluation"]["lambda_validation_specs"][2])

    validation_data_loader_dict = {}

    for k in kappa_val_data:
        for l in lambda_val_data:

            DS_i= DataSetScalarTheory2D_preprocessed_data(
                N = config["config_data"]["N"],
                mode = "validation",
                kappa_list = [round(k,4)],
                lambda_list = [round(l,4)],
                augment=True,
                max_samples=args_dict["n_samples_val_plain"],
                sigma_noise=0.0,
                base_path="./data/ScalarTheory/validation_data/"
            )
            
            DL_i = DataLoader(
                DS_i,
                batch_size = config["config_evaluation"]["batch_size_validation"],
                shuffle = True,
                num_workers = 4
            )

            validation_data_loader_dict[f"k={round(k,4)}_l={round(l,4)}"] = DL_i

    with torch.no_grad():

        ##############################################################################
        # Compute the validation nll for each parameter set
        ##############################################################################

        validation_nll_dict = {
            "nll":{},
            "error":{}
        }

        for k in tqdm.tqdm(kappa_val_data):
            for l in lambda_val_data:

                DL_i = validation_data_loader_dict[f"k={round(k,4)}_l={round(l,4)}"]  

                log_p_theta_val = torch.zeros([0])

                for j,(kappa_batch,lambda_batch,x_batch) in enumerate(DL_i):   

                    log_p_theta_val_j = INN.log_prob(x = x_batch.to(device),beta_tensor=kappa_batch.to(device)).detach().cpu()
                    log_p_theta_val = torch.cat((log_p_theta_val,log_p_theta_val_j),0)

                # Perform bootstrapping to estimate the error of the nll
                samples_nll = np.zeros(args_dict["n_samples_bootstrap"])
            
                for i in range(args_dict["n_samples_bootstrap"]):
                    indices = np.random.randint(0,len(log_p_theta_val),len(log_p_theta_val))
                    samples_nll[i] =  - log_p_theta_val[indices].mean()
            
                mean_samples_nll = samples_nll.mean()

                error_nll_j = np.sqrt(np.square(samples_nll - mean_samples_nll).sum() / (args_dict["n_samples_bootstrap"] - 1))
                
                val_nll_j =  - log_p_theta_val.mean().item()
               
                validation_nll_dict["nll"][f"k={round(k,4)}_l={round(l,4)}"] = val_nll_j
                validation_nll_dict["error"][f"k={round(k,4)}_l={round(l,4)}"] = error_nll_j
                
        # Save the results
        with open(os.path.join(results_folder,"validation_nll_dict.json"),"w") as f:
            json.dump(validation_nll_dict,f)
        f.close()

        ##############################################################################
        # Compute relative effective sample size for each parameter
        ##############################################################################

        kappa_ESS = np.linspace(args_dict["min_kappa_ESS"],args_dict["max_kappa_ESS"],args_dict["n_kappa_ESS"])
        lambda_ESS = [0.02]
        
        n_batches_ESS = int(args_dict["n_samples_ESS"] / args_dict["batch_size_ESS"])

        ESS_storage = torch.zeros([len(kappa_ESS),3])

        for counter,k in tqdm.tqdm(enumerate(kappa_ESS)):
            for l in lambda_ESS:

                log_p_target_INN_ESS = torch.zeros([0])
                log_p_theta_INN_ESS = torch.zeros([0])

                for j in range(n_batches_ESS):

                    #Compute the log likelihood of the INN samples
                    samples_j = INN.sample(n_samples = args_dict["batch_size_ESS"],beta_tensor = k)

                    log_p_theta_INN_j = INN.log_prob(samples_j,k).cpu()
                    log_p_theta_INN_ESS = torch.cat((log_p_theta_INN_ESS,log_p_theta_INN_j.detach().cpu()),0)

                    log_p_target_INN_j = - S_ScalarTheory(samples_j.cpu(),kappas = k,lambdas = l).cpu()
                    log_p_target_INN_ESS = torch.cat((log_p_target_INN_ESS,log_p_target_INN_j.detach().cpu()),0)

                # Get the error of the ESS by bootstrapping
                samples = np.zeros(args_dict["n_samples_bootstrap"])

                for i in range(args_dict["n_samples_bootstrap"]):
                    indices = np.random.randint(0,len(log_p_theta_INN_ESS),len(log_p_theta_INN_ESS))
            
                    samples[i] =  get_ESS_r(
                        log_p_target_INN=log_p_target_INN_ESS[indices],
                        log_p_theta_INN=log_p_theta_INN_ESS[indices]
                    )

                mean_samples = samples.mean()
                error_ESS_kl = np.sqrt(np.square(samples - mean_samples).sum() / (args_dict["n_samples_bootstrap"] - 1))

                ESS_r_kl = get_ESS_r(
                        log_p_target_INN=log_p_target_INN_ESS,
                        log_p_theta_INN=log_p_theta_INN_ESS
                    )

                ESS_storage[counter][0] = k
                ESS_storage[counter][1] = ESS_r_kl
                ESS_storage[counter][2] = error_ESS_kl
        
        # Save the results
        np.savetxt(
            fname = os.path.join(results_folder,"ESS_r.txt"),
            X = ESS_storage,
            header="kappa\tESS_r(kappa)\terror_ESS_r(kappa)"
        )

        ##############################################################################
        # Compute the physical observables
        ##############################################################################

        kappa_physics = np.linspace(args_dict["min_kappa_physics"],args_dict["max_kappa_physics"],args_dict["n_kappa_physics"])
        lambda_physics = [0.02]

        if len(lambda_physics) > 1:
            raise NotImplementedError
            
        output_file_physics = os.path.join(results_folder,f"summary_phyiscs_properties_lambda_{lambda_physics[0]}.txt")
  
        n_batches_phyiscs = int(args_dict["n_samples_physics"] / args_dict["batch_size_physics"])

        magnetizations_i = torch.zeros([len(kappa_physics),2])
        actions_gt_i = torch.zeros([len(kappa_physics),2])
        susceptibility_i = torch.zeros([len(kappa_physics),2])
        binder_cumulant_i = torch.zeros([len(kappa_physics),2])

        counter_physics = 0

        for k in tqdm.tqdm(kappa_physics):
            for l in lambda_physics:

                actions_gt_kl_u = torch.zeros([0])
                magnetization_kl_u = torch.zeros([0])

                for i in range(n_batches_phyiscs):
                    with torch.no_grad():
                        samples_kli_u = INN.sample(n_samples = args_dict["batch_size_physics"],beta_tensor = k).cpu().detach()

                        action_gt_kli_u = S_ScalarTheory(samples_kli_u,k,l)
                        magnetization_kli_u = samples_kli_u.sum(dim = (1,2,3))

                    actions_gt_kl_u = torch.cat((actions_gt_kl_u,action_gt_kli_u),0)
                    magnetization_kl_u = torch.cat((magnetization_kl_u,magnetization_kli_u),0)

                N = config["config_data"]["N"]

                mean_magnetization,std_magnetization = bootstrap(x = np.abs(np.array(magnetization_kl_u)) / N**2,s = np.mean,args={"axis":0})
                mean_action_gt,std_action_gt = bootstrap(x = np.array(actions_gt_kl_u) / N**2,s = np.mean,args={"axis":0})
                susceptibility_mean,sigma_susceptibility = bootstrap(x = np.abs(np.array(magnetization_kl_u)),s = get_susceptibility,args={"Omega":N**2})
                U_L_mean,sigma_U_L = bootstrap(x = np.array(magnetization_kl_u),s = get_U_L,args={"Omega":N**2})

                magnetizations_i[counter_physics] = torch.Tensor([mean_magnetization,std_magnetization])
                actions_gt_i[counter_physics] = torch.Tensor([mean_action_gt,std_action_gt])
                susceptibility_i[counter_physics] = torch.Tensor([susceptibility_mean,sigma_susceptibility])
                binder_cumulant_i[counter_physics] = torch.Tensor([U_L_mean,sigma_U_L])

                counter_physics += 1

        # Save the results as a text file. Write the infirmation in the following order: kappa, magnetization, action, susceptibility, binder cumulant and add it to the header
        header = "kappa,mean_magnetization,std_magnetization,mean_action_gt,std_action_gt,susceptibility_mean,sigma_susceptibility,U_L_mean,sigma_U_L"
        data = np.concatenate((kappa_physics[:,np.newaxis],magnetizations_i,actions_gt_i,susceptibility_i,binder_cumulant_i),axis = 1)
        np.savetxt(output_file_physics,data,header = header,comments = "")

        ##############################################################################
        # Compute the distribution of spin values
        ##############################################################################

        kappas_spins_INN = np.arange(args_dict["min_kappa_spins_INN"],args_dict["max_kappa_spins_INN"],args_dict["d_kappa_spins_INN"])
        kappas_spins_val = np.arange(args_dict["min_kappa_spins_val"],args_dict["max_kappa_spins_val"],args_dict["d_kappa_spins_val"])
        lambda_spins = [0.02]

        # Spin distribution of the validation data
        storage_spins_val = None

        header ="spin_val"

        for k in kappas_spins_val:
            for l in lambda_spins:

                DL_kl = validation_data_loader_dict[f"k={round(k,4)}_l={round(l,4)}"]

                densities_kl_val,bins_kl_val = torch.histogram(
                    DL_kl.dataset.data.reshape(-1),
                    density=True,
                    bins = args_dict["bins_spin_dist_hist"],
                    range=[-args_dict["max_abs_spin_val_hist"],args_dict["max_abs_spin_val_hist"]]
                )

                if storage_spins_val is None:
                    storage_spins_val = bins_kl_val.reshape(-1,1)

                densities_kl_val = torch.cat((densities_kl_val.reshape(-1,1),torch.Tensor([[-1]])),0)
                storage_spins_val = torch.hstack((storage_spins_val,densities_kl_val))

                header += f"\t{round(k,4)}"

                n_samples = len(DL_kl.dataset)

        np.savetxt(
            fname = os.path.join(results_folder,"spins_validation_data.txt"),
            X = storage_spins_val.numpy(),
            header=header
        )

        # Spin distribution of the learned distribution
        header ="spin_INN"
        storage_spins_INN = None

        for k in kappas_spins_INN:
            for l in lambda_spins:

                n_batches_spin_INN = int(n_samples / args_dict["batch_size_INN_spins"])

                spins_KL_INN = torch.zeros([0])

                for abc in range(n_batches_spin_INN):
                    with torch.no_grad():
                        samples_kli = INN.sample(n_samples = args_dict["batch_size_INN_spins"],beta_tensor = k).cpu().detach()
                        spins_KL_INN = torch.cat((spins_KL_INN,samples_kli.flatten()),0)

                densities_kl_INN,bins_kl_INN = torch.histogram(
                    spins_KL_INN,
                    density=True,
                    bins = args_dict["bins_spin_dist_hist"],
                    range=[-args_dict["max_abs_spin_val_hist"],args_dict["max_abs_spin_val_hist"]]
                )

                if storage_spins_INN is None:
                    storage_spins_INN = bins_kl_INN.reshape(-1,1)

                densities_kl_INN = torch.cat((densities_kl_INN.reshape(-1,1),torch.Tensor([[-1]])),0)
                storage_spins_INN = torch.hstack((storage_spins_INN,densities_kl_INN))

                header += f"\t{round(k,4)}"

        np.savetxt(
            fname = os.path.join(results_folder,"spins_INN_data.txt"),
            X = storage_spins_INN.numpy(),
            header=header
        )

if __name__ == "__main__":

    # Load the model
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_folder',              type = str,   required=False, default=None,     help = "Location of the experiment to evaluate.")
    parser.add_argument('--experiment_parent_folder',       type = str,   required=False, default=None,     help = "Folder with multiple experiments to evaluate.")
    parser.add_argument('--use_last',                       type = bool,  required=False, default=False,    help = "Use model observed at the end of the training run.")
    parser.add_argument('--n_samples_val_plain',            type = int,   required=False, default=5000,     help = "Number of samples taken from the validation set (Before applying data augmentation).")
    parser.add_argument('--n_samples_bootstrap',            type = int,   required=False, default=20,       help = "Number of bootstrap samples.")

    # Evaluation of the effective sample size
    parser.add_argument('--n_kappa_ESS',          type = float, required=False, default=200,    help = "Number of kappa values for which the ESS is evaluated.")
    parser.add_argument('--min_kappa_ESS',        type = float, required=False, default=0.22,   help = "Smallest kappa value for which the ESS is evaluated.")
    parser.add_argument('--max_kappa_ESS',        type = float, required=False, default=0.32,   help = "Largest kappa value for which the ESS is evaluated.")
    parser.add_argument('--n_samples_ESS',        type = int,   required=False, default=10000,  help = "Number of samples used to compute the ESS.")
    parser.add_argument('--batch_size_ESS',       type = int,   required=False, default=2000,   help = "Batch size for the computation of the ESS.")

    # Evaluation of the physical observables
    parser.add_argument('--n_kappa_physics',            type = float, required=False, default=200,      help = "Number of kappa values for which the phsical observables are evaluated.")
    parser.add_argument('--min_kappa_physics',          type = float, required=False, default=0.22,     help = "Smallest kappa value for which the phsical observables is evaluated.")
    parser.add_argument('--max_kappa_physics',          type = float, required=False, default=0.32,     help = "Largest kappa value for which the phsical observables is evaluated.")
    parser.add_argument('--n_samples_physics',          type = int,   required=False, default=10000,    help = "Number of samples used to compute the physical observables.")
    parser.add_argument('--batch_size_physics',         type = int,   required=False, default=2000,     help = "Batch size for the computation of the physical observables.")

    # Evaluation of the spin distributions
    parser.add_argument('--min_kappa_spins_INN',        type = float,   required=False, default=0.22,   help = "Smalles kappa in the evaluation of the spin distribution for the INN.")
    parser.add_argument('--max_kappa_spins_INN',        type = float,   required=False, default=0.32,   help = "Largest kappa in the evaluation of the spin distribution for the INN.")
    parser.add_argument('--d_kappa_spins_INN',          type = float,   required=False, default=0.01,   help = "Step size for kappa in the evaluation of the spin distribution for the INN.")
    parser.add_argument('--min_kappa_spins_val',        type = float,   required=False, default=0.22,   help = "Smalles kappa in the evaluation of the spin distribution for the validation data.")
    parser.add_argument('--max_kappa_spins_val',        type = float,   required=False, default=0.32,   help = "Largest kappa in the evaluation of the spin distribution for the validation data.")
    parser.add_argument('--d_kappa_spins_val',          type = float,   required=False, default=0.01,   help = "Step size for kappa in the evaluation of the spin distribution for the validation data.")
    parser.add_argument('--bins_spin_dist_hist',        type = int,     required=False, default=100,    help = "Number of bins in the histograms.")
    parser.add_argument('--max_abs_spin_val_hist',      type = float,   required=False, default=5.0,    help = "Largest absolute spin value depicted in the plot.")
    parser.add_argument('--batch_size_INN_spins',       type = int,     required=False, default=1000,   help = "Batch size for sampling from the INN.")

    args = parser.parse_args()

    meta_data_dict = vars(args)

    # One experiment
    if (meta_data_dict["experiment_parent_folder"] is None) and (meta_data_dict["experiment_folder"] is not None):
        run_evaluation(experiment_folder=meta_data_dict["experiment_folder"],args_dict=meta_data_dict)
    
    # Loop over every folder in the collection of experiments
    elif (meta_data_dict["experiment_parent_folder"] is not None) and (meta_data_dict["experiment_folder"] is None):

        # Get all the experiments in the parent folder
        experiment_folders = os.listdir(meta_data_dict["experiment_parent_folder"])

        for experiment_folder in experiment_folders:

            try:
                folder_i = os.path.join(meta_data_dict["experiment_parent_folder"],experiment_folder)
                print(f"Process experiment {folder_i}")

                run_evaluation(experiment_folder=folder_i,args_dict=meta_data_dict)
            
            except Exception as e:
                print("An exception occurred:", e)

    else:
        raise ValueError("Select folder containing the experiments.")


        