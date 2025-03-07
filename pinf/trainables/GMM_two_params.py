from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import partial
import torch

from pinf.trainables.base import BaseTrainableObject_MultipleExternalParameters
from pinf.plot.utils import eval_pdf_on_grid_2D
from pinf.datasets.log_likelihoods import log_p_2D_ToyExample_two_parameters
from pinf.models.INN import INN_Model
from pinf.datasets.datasets import DataSet_2D_ToyExample_external_two_parameters

class TrainingObject_2D_ToyExample_two_external_parameters(BaseTrainableObject_MultipleExternalParameters):
    def __init__(self,INN:INN_Model,config:dict)->None:
        """
        Base class for training of a Normalizing flow conditioned one multiple external parameters.

        parameters:
            INN:    Normalizing flow to train
            config: Configuration file
        """


        # Initialize configs for the data free loss model
        if config["config_training"]["data_free_loss_mode"] == "reverse_KL_multi_param":
            config["config_training"]["log_p_target_kwargs"] = {}

        elif config["config_training"]["data_free_loss_mode"] == "TRADE_no_grid_multi_param":

            config["config_training"]["loss_model_params"]["S_kwargs"] = {"device":config["device"]}

            config["config_training"]["loss_model_params"]["dSdparam_kwargs_list"] = []

            for dS_dparam_name in config["config_training"]["dS_dparam_names"]:
                config["config_training"]["loss_model_params"]["dSdparam_kwargs_list"].append({"device":config["device"]})
                config["config_training"]["loss_model_params"]["base_parameter_list"] = config["config_data"]["init_data_set_params"]["parameter_coordinates"]
            

        super().__init__(INN, config)

        self.validation_data_loader_dict = {}

        # Load validation data
        for alpha in config["config_evaluation"]["alpha_list_validation"]:
            for beta in config["config_evaluation"]["beta_list_validation"]:

                DS_ij = DataSet_2D_ToyExample_external_two_parameters(
                    d = 2,
                    parameter_coordinates = [[alpha,beta]],
                    mode="validation",
                    n_samples=config["config_evaluation"]["samples_validation_set"]
                )

                DL_ij = DataLoader(
                    DS_ij, 
                    batch_size=config["config_evaluation"]["batch_size_validation"], 
                    shuffle=True,
                    num_workers=11,
                    )
                
                self.validation_data_loader_dict[f"alpha_{alpha}_beta_{beta}"] = DL_ij

    def on_train_epoch_end(self):
        """
        Plotting and model evaluation
        """

        super().on_train_epoch_end()

        if (((self.current_epoch + 1) % self.config["config_evaluation"]["plot_freq"] == 0) or (self.current_epoch + 1) == self.config_training["n_epochs"] or self.current_epoch == 0):
            
            # Plot the densities of the trained model and the ground truth
            try:
                self.INN.train(False)
                
                ################################################################################################################################################
                # Plot the ground truth distribution
                ################################################################################################################################################
                fig,axes = plt.subplots(len(self.config["config_evaluation"]["alpha_list_validation"]),len(self.config["config_evaluation"]["beta_list_validation"]),figsize = (5 * len(self.config["config_evaluation"]["beta_list_validation"]),5 * len(self.config["config_evaluation"]["alpha_list_validation"])))

                for i,alpha in enumerate(self.config["config_evaluation"]["alpha_list_validation"]):
                    for j,beta in enumerate(self.config["config_evaluation"]["beta_list_validation"]):
                        
                        log_p_ij = partial(log_p_2D_ToyExample_two_parameters,parameter_list = [alpha,beta],device = self.device)

                        pdf_grid,x_grid,y_grid = eval_pdf_on_grid_2D(
                            pdf = log_p_ij,
                            x_lims = self.config["config_evaluation"]["grid_lim_list"][0],
                            y_lims = self.config["config_evaluation"]["grid_lim_list"][1],
                            x_res = self.config["config_evaluation"]["grid_res_list"][0],
                            y_res = self.config["config_evaluation"]["grid_res_list"][1],
                            device=self.device
                            )
                    
                        axes[i][j].imshow(pdf_grid.cpu().detach().exp().numpy(),extent = [x_grid.min(),x_grid.max(),y_grid.min(),y_grid.max()],origin = 'lower',cmap = "jet")

                        axes[i][j].set_title(f'a = {alpha}, b = {round(beta,5)}')
                        axes[i][j].axis('off')
                
                plt.tight_layout()
                self.logger.experiment.add_figure(f'ground_truth_density', fig, self.current_epoch + 1)
                plt.close(fig)

                ################################################################################################################################################
                # Plot the INN distribution
                ################################################################################################################################################

                fig,axes = plt.subplots(len(self.config["config_evaluation"]["alpha_list_validation"]),len(self.config["config_evaluation"]["beta_list_validation"]),figsize = (5 * len(self.config["config_evaluation"]["beta_list_validation"]),5 * len(self.config["config_evaluation"]["alpha_list_validation"])))

                for i,alpha in enumerate(self.config["config_evaluation"]["alpha_list_validation"]):
                    for j,beta in enumerate(self.config["config_evaluation"]["beta_list_validation"]):
                        
                        with torch.no_grad():
                            log_p_ij = partial(self.INN.log_prob,parameter_list = [alpha,beta])

                            pdf_grid,x_grid,y_grid = eval_pdf_on_grid_2D(
                                pdf = log_p_ij,
                                x_lims = self.config["config_evaluation"]["grid_lim_list"][0],
                                y_lims = self.config["config_evaluation"]["grid_lim_list"][1],
                                x_res = self.config["config_evaluation"]["grid_res_list"][0],
                                y_res = self.config["config_evaluation"]["grid_res_list"][1],
                                device=self.device
                                )
                        
                            axes[i][j].imshow(pdf_grid.cpu().detach().exp().numpy(),extent = [x_grid.min(),x_grid.max(),y_grid.min(),y_grid.max()],origin = 'lower',cmap = "jet")

                            axes[i][j].set_title(f'a = {alpha}, b = {round(beta,5)}')
                            axes[i][j].axis('off')
                
                plt.tight_layout()
                self.logger.experiment.add_figure(f'INN_density', fig, self.current_epoch + 1)
                plt.close(fig)

                self.INN.train(True)
        
            except Exception as e:
                print("When plotting the densities of the trained model and the ground truth, the following error occured:")
                print(e)
