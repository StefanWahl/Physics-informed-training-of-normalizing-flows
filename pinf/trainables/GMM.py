import torch
import numpy as np
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import os

from pinf.trainables.base import BaseTrainableObject_TemperatureScaling
from pinf.models.INN import INN_Model
from pinf.models.GMM import GMM
from pinf.datasets.energies import S_2D_GMM
from pinf.datasets.log_likelihoods import log_p_2D_GMM
from pinf.datasets.datasets import DataSet2DGMM
from pinf.trainables.utils import save_data
from pinf.plot.utils import (
    eval_pdf_on_grid_2D,
    plot_pdf_2D
)

# Set parameters for the GMM
means = torch.tensor([
    [-1.0,2.0],
    [3.0,7.0],
    [-4.0,2.0],
    [-2.0,-4.0],
    [0.0,4.0],
    [5.0,-2.0]
])

#Covariance matrices
S = torch.tensor([
        [[ 0.2778,  0.4797],
        [ 0.4797,  0.8615]],

        [[ 0.8958, -0.0249],
        [-0.0249,  0.1001]],

        [[ 1.3074,  0.9223],
        [ 0.9223,  0.7744]],

        [[ 0.0305,  0.0142],
        [ 0.0142,  0.4409]],

        [[ 0.0463,  0.0294],
        [ 0.0294,  0.3441]],
        
        [[ 0.15,  0.0294],
        [ 0.0294,  1.5]]])

class TrainingObject_2D_GMM(BaseTrainableObject_TemperatureScaling):
    def __init__(self,INN:INN_Model,config:dict)->None:
        """
        Trainable for the power-scaled two-dimensional Gaussian Mixture Model

        parameters:
            INN:    Normalizing flow to train
            config: Configuration file
        """

        # Initialize the target distribution
        gmm = GMM(means = means,covs=S,device=config["device"])
        self.gmm = gmm

        # Set parameters
        config["config_evaluation"]["log_p_target_kwargs"] = {"gmm":gmm}

        if "log_p_target_name" in config["config_training"].keys():
            config["config_training"]["log_p_target_kwargs"] = {"gmm":gmm}

        if "data_free_loss_mode" in config["config_training"].keys() and ((config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V3") or (config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V2") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V3") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V2") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param")):
            config["config_training"]["loss_model_params"]["S_kwargs"] = {"gmm":gmm,"device": config["device"]}
            config["config_training"]["loss_model_params"]["dSdparam_kwargs"] = {"gmm":gmm,"device": config["device"]}

        if ("use_reweighted_nll_loss" in config["config_training"]) and (config["config_training"]["use_reweighted_nll_loss"]):
            self.S_kwargs = {"gmm":gmm,"device": config["device"]}
            self.S = S_2D_GMM

        # Initialize parent
        super(TrainingObject_2D_GMM, self).__init__(INN=INN,config = config)

        # Get validation parameters
        T_list = torch.linspace(np.log(0.1),np.log(10),20).exp()
        T_list = torch.cat((T_list,torch.tensor([1.0])))
        T_list = [round(T_list.sort().values[i].item(),5) for i in range(len(T_list))]

        T_list_eval = T_list[10 - config["config_evaluation"]["n_validation_temp_left_right"]:-(10 - config["config_evaluation"]["n_validation_temp_left_right"])]

        self.validation_data_temperatures = torch.zeros(len(T_list_eval))

        # Load the validation data
        self.validation_data_loader_dict = {}

        for i,T_i in enumerate(T_list_eval):

            T_i = round(T_i,5)

            DS_i = DataSet2DGMM(
                d = config["config_data"]["init_data_set_params"]["d"],
                mode = "validation",
                temperature_list=[T_i],
                base_path=config["config_data"]["init_data_set_params"]["base_path"],
                n_samples=config["config_evaluation"]["samples_validation_set"]
                )

            DL_i = DataLoader(
                DS_i,
                batch_size = self.config["config_evaluation"]["batch_size_validation"],
                shuffle = True,
                num_workers = 4
            )

            self.validation_data_temperatures[i] = T_i
            self.validation_data_loader_dict[f"{T_i}"] = DL_i

        self.validation_data_temperatures = torch.round(input = self.validation_data_temperatures,decimals=5)
        print(f"Validation temperatures: {self.validation_data_temperatures}")

        # Load the approximated partition functions
        with open("./data/2D_GMM/Z_T.json","r") as f:
            self.Z_T_dict = json.load(f)
        f.close()

        # Load training data if training samples are used as proposal states for the computation of the physics-informed loss contribution
        if ("proposal_distribution_type" in self.config_training.keys()) and (self.config_training["proposal_distribution_type"] == "training_data"): 
            print("Load training data for sampling of proposal states ")

            self.DS_training = DataSet2DGMM(
                d = config["config_data"]["init_data_set_params"]["d"],
                mode = "training",
                temperature_list=[self.base_betas.item()],
                base_path=config["config_data"]["init_data_set_params"]["base_path"],
                n_samples=1e7
                )

    def validation(self):
        """
        Evaluate the model performance on the validation set.
        """

        if ((self.current_epoch + 1) % self.config["config_evaluation"]["validation_freq"] == 0) or (self.current_epoch == 0) or (self.current_epoch + 1 == self.config_training["n_epochs"]):

                KL_list = []

                with torch.no_grad():
                    self.INN.train(False)

                    for T_i in self.validation_data_temperatures:

                        T_i = round(T_i.item(),5)

                        DL_i = self.validation_data_loader_dict[f"{T_i}"]

                        log_p_theta_val = torch.zeros([0])

                        for j,(beta_batch,x_batch) in enumerate(DL_i):

                            log_p_theta_val_i = self.INN.log_prob(x_batch.to(self.config["device"]),beta_tensor=beta_batch.to(self.config["device"]))
                            log_p_theta_val = torch.cat((log_p_theta_val,log_p_theta_val_i.detach().cpu()),0)
                            

                        #Get the average validation negative log-likelihood
                        nll_i = - log_p_theta_val.mean().item()
                        KL_list.append(nll_i)

                self.current_mean_KL = np.mean(KL_list)

                self.log_dict({f"model_performance/mean_validation_KL":self.current_mean_KL})

                self.INN.train(True)

    def on_train_epoch_end(self):
        """
        Perform evaluations of the model
        """

        # Evaluate the model performance
        self.validation()

        if (((self.current_epoch + 1) % self.config["config_evaluation"]["plot_freq"] == 0) or (self.current_epoch + 1) == self.config_training["n_epochs"] or self.current_epoch == 0) and self.config["config_evaluation"]["run_evaluations"]:
            
            ######################################################################################################################################################
            #Plot running average of the internal expectation values
            ######################################################################################################################################################

            # TRADE grid loss
            if (self.config_training["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V2") or (self.config_training["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V3"):
                
                # Expectation value
                fig,ax = plt.subplots(1,1,figsize = (10,5))
                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),self.data_free_loss_model.EX_A.detach().cpu().numpy(),color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel(r"$\langle A \rangle$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/EX_A', fig, self.current_epoch + 1)
                plt.close(fig)

                # Data-free loss contribution
                fig,ax = plt.subplots(1,1,figsize = (10,5))

                loss_plot = self.data_free_loss_model.loss_statistics.detach().cpu()
                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),loss_plot,color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel(r"$\langle L(\beta) \rangle$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/loss_per_bin', fig, self.current_epoch + 1)
                plt.close(fig)

                # Log-causality weights
                fig,ax = plt.subplots(1,1,figsize = (10,5))

                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),self.data_free_loss_model.log_causality_weights.detach().cpu(),color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel(r"$\log{\omega(\beta)}$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/log_importance_weights', fig, self.current_epoch + 1)
                plt.close(fig)

                # Causality weights
                fig,ax = plt.subplots(1,1,figsize = (10,5))

                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),self.data_free_loss_model.log_causality_weights.detach().cpu().exp()/self.data_free_loss_model.log_causality_weights.detach().cpu().exp().sum(),color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel(r"$\omega(\beta)$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/importance_weights', fig, self.current_epoch + 1)
                plt.close(fig)

                # Save the values in the dictionary where the training progress is stored
                base_path_data = os.path.join(self.logger.log_dir,"recorded_data")
                
                # Initialize the dictionary if it does not exist
                if not os.path.exists(base_path_data):
                    os.makedirs(base_path_data)

                # Save the data
                save_data(
                    file_path=os.path.join(base_path_data,"loss_PI.txt"),
                    x = self.data_free_loss_model.param_grid.squeeze().cpu().detach().numpy(),
                    y_new = self.data_free_loss_model.loss_statistics.detach().cpu().detach().numpy(),
                    epoch = self.current_epoch,
                    header = "kappa\tgradient loss e1\tgradient loss e2\t..."
                    )
                
                save_data(
                    file_path=os.path.join(base_path_data,"log_causality_weights.txt"),
                    x = self.data_free_loss_model.param_grid.squeeze().cpu().detach().numpy(),
                    y_new = self.data_free_loss_model.log_causality_weights.detach().cpu().detach().numpy(),
                    epoch = self.current_epoch,
                    header = "kappa\tlog causality weights e1\tlog causality weights e2\t..."
                    )
                
                save_data(
                    file_path=os.path.join(base_path_data,"EX_A.txt"),
                    x = self.data_free_loss_model.param_grid.squeeze().cpu().detach().numpy(),
                    y_new = self.data_free_loss_model.EX_A.detach().cpu().squeeze().numpy(),
                    epoch = self.current_epoch,
                    header = "kappa\tEX_A e1\tEX_A e2\t..."
                    )

            # TRADE without grid
            elif (self.config_training["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V2") or (self.config_training["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V3"):

                # Expectation values
                mask = (self.data_free_loss_model.param_storage_grid.squeeze().cpu() != 0)
                fig,ax = plt.subplots(1,1,figsize = (10,5))
                ax.plot(self.data_free_loss_model.param_storage_grid.squeeze().cpu()[mask],self.data_free_loss_model.EX_storage_grid.detach().cpu().numpy()[mask],color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel(r"$\langle A \rangle$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/EX_A', fig, self.current_epoch + 1)
                plt.close(fig)

                # Save the values in the dictionary where the training progress is stored
                base_path_data = os.path.join(self.logger.log_dir,"recorded_data")
                
                # Initialize the dictionary if it does not exist
                if not os.path.exists(base_path_data):
                    os.makedirs(base_path_data)

                # Save the data
                save_data(
                    file_path=os.path.join(base_path_data,"EX_A.txt"),
                    x = self.data_free_loss_model.param_storage_grid.squeeze().cpu().detach().numpy(),
                    y_new = self.data_free_loss_model.EX_storage_grid.detach().cpu().squeeze().numpy(),
                    epoch = self.current_epoch,
                    header = "kappa\tEX_A e1\tEX_A e2\t..."
                    )
                
            ######################################################################################################################################################
            #Plot densities
            ######################################################################################################################################################
            with torch.no_grad():
                self.INN.train(False)

                label_densities = ["GT","log GT","INN","log INN"]

                range_val = None

                fig_densities, ax_densities = plt.subplots(len(self.validation_data_temperatures),4,figsize=(4 * 5,len(self.validation_data_temperatures) * 5))

                for i,T_i in enumerate(self.validation_data_temperatures):
                    T_i = round(T_i.item(),5)  

                    lim_list_grid=self.config["config_evaluation"]["grid_lim_list"]
                    res_list_grid=self.config["config_evaluation"]["grid_res_list"]

                    # Evaluate the pdfs on a grid
                    log_p_gt_grid,x_grid,y_grid = eval_pdf_on_grid_2D(
                        pdf = log_p_2D_GMM,
                        x_lims = lim_list_grid[0],
                        y_lims = lim_list_grid[1],
                        x_res = res_list_grid[0],
                        y_res = res_list_grid[1],
                        device = self.config["device"],
                        kwargs_pdf={"beta_tensor":1 / T_i,"gmm":self.gmm,"device":self.config["device"]}
                        )
                    
                    log_p_INN_grid,x_grid,y_grid = eval_pdf_on_grid_2D(
                        pdf = self.INN.log_prob,
                        x_lims = lim_list_grid[0],
                        y_lims = lim_list_grid[1],
                        x_res = res_list_grid[0],
                        y_res = res_list_grid[1],
                        device = self.config["device"],
                        kwargs_pdf={"beta_tensor":1 / T_i}
                        )
                    
                    log_p_gt_grid = log_p_gt_grid - np.log(self.Z_T_dict[str(T_i)])

                    min_log_range = min(log_p_gt_grid.min().item(),log_p_INN_grid.min().item())
                    max_log_range = max(log_p_gt_grid.max().item(),log_p_INN_grid.max().item())

                    # Plot the densities and the log-densities
                    for j,grid in enumerate([log_p_gt_grid.exp(),log_p_gt_grid,log_p_INN_grid.exp(),log_p_INN_grid]):
                        
                        if j == 0 or j == 2: range_val = [np.exp(min_log_range),np.exp(max_log_range)]
                        else: range_val = [min_log_range,max_log_range]

                        im_j = plot_pdf_2D(
                            pdf_grid=grid.reshape(self.config["config_evaluation"]["grid_res_list"][0],self.config["config_evaluation"]["grid_res_list"][1]).cpu().detach(),
                            x_grid = x_grid.cpu().detach(),
                            y_grid = y_grid.cpu().detach(),
                            ax = ax_densities[i,j],
                            title = f"Temperature: {T_i}\n {label_densities[j]}",
                            turn_off_axes=True,
                            range_vals = range_val,
                            return_im=True,
                            cmap = "jet"
                        )

                        # Add colorbar
                        fig_densities.colorbar(mappable = im_j, ax=ax_densities[i,j])

                plt.tight_layout()

                # Add the plots to the tensorboard
                self.logger.experiment.add_figure('Densities', fig_densities, self.current_epoch + 1)
                plt.close(fig_densities)

                self.INN.train(True)

    def get_evaluation_points(self,beta_tensor:torch.Tensor)->torch.Tensor:
        """
        Sample points for the evaluation of the physics-informed loss contribution

        parameters:
            beta_tensor:    Condition values where an evaluation point should be sampled

        returns:
            x:              Batch of points following the model distribution at the specified condition values
        """

        if "proposal_distribution_type" in self.config_training.keys():

            # Use model at base condition as proposal distribution
            if self.config_training["proposal_distribution_type"] == "model_at_base_param":
                with torch.no_grad():
                    x = self.INN.sample(n_samples = len(beta_tensor),beta_tensor = self.base_betas.item())
                return x.detach()

            # Use training samples as proposal states
            elif self.config_training["proposal_distribution_type"] == "training_data":
                idx = torch.randperm(len(self.DS_training.data))[:len(beta_tensor)]
                x_idx = self.DS_training.data[idx]
                return x_idx.to(beta_tensor.device)

            # Use the conditional model as proposal distribution
            elif self.config_training["proposal_distribution_type"] == "model":
                with torch.no_grad():
                    x = self.INN.sample(n_samples = len(beta_tensor),beta_tensor = beta_tensor)
                return x.detach()
            
            # Use a standard normal distribution as proposal distribution
            elif self.config_training["proposal_distribution_type"] == "standard_normal":
                x = torch.randn([len(beta_tensor),2]).to(beta_tensor.device)
                return x
            
            else:
                raise ValueError()

        # Default: Use the conditional model distribution
        else:
            with torch.no_grad():
                x = self.INN.sample(n_samples = len(beta_tensor),beta_tensor = beta_tensor)

            return x.detach()
