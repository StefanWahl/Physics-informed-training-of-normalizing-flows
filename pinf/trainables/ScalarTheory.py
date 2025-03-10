import torch
import torchvision.transforms as Trafo
from functools import partial
import numpy as np
import lightning as L
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tbparse import SummaryReader
import importlib.util
import sys  
import os

from pinf.trainables.utils import (
    optimizer_dict,
    MultiCycleLR,
    remove_non_serializable
    )

from pinf.datasets.transformations import (
    RandomChangeOfSign,
    RandomHorizentalRoll,
    RandomVerticalRoll
)

from pinf.datasets.datasets import DataSetScalarTheory2D_preprocessed_data

from pinf.losses.factory import DataFreeLossFactory

from pinf.models.INN import INN_Model

from pinf.trainables.utils import save_data

from pinf.datasets.energies import S_ScalarTheory

from pinf.datasets.log_likelihoods import log_p_target_dict

from pinf.plot.utils import (
    bootstrap,
    get_U_L,
    get_susceptibility
)

class TrainingObject_2D_Scalar_Theory(L.LightningModule):
    def __init__(self,INN:INN_Model,config:dict)->None:
        """
        parameters:
            INN:    Normalizing flow to train
            config: Configuration file
        """

        super(TrainingObject_2D_Scalar_Theory, self).__init__()

        # Transformation for the data augmentation
        self.transformation = Trafo.Compose([
            Trafo.RandomHorizontalFlip(p=0.5),
            Trafo.RandomVerticalFlip(p=0.5),
            RandomChangeOfSign(p=0.5),
            RandomHorizentalRoll(),
            RandomVerticalRoll()
            ])
        
        if config["config_training"]["data_free_loss_mode"]is not None:
        
            self.fixed_relative_weighting = config["config_training"]["fixed_relative_weighting"]

            self.flag_adaptive_weighting_nll = False

            if (config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V2") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param") or (config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V2") or (config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_advanced_grid"):
                config["config_training"]["loss_model_params"]["S_kwargs"] = {"lambdas":config["config_data"]["init_data_set_params"]["lambda_list"][0]}
                config["config_training"]["loss_model_params"]["dSdparam_kwargs"] = {}
                config["config_training"]["loss_model_params"]["base_params"] = torch.tensor(config["config_data"]["init_data_set_params"]["kappa_list"])
                config["config_training"]["loss_model_params"]["n_epochs"] = config["config_training"]["n_epochs"] - config["config_training"]["regularization_data_free_start"]
                config["config_training"]["loss_model_params"]["n_batches_per_epoch"] = config["config_training"]["n_batches_per_epoch"]
            
            if "alpha_adaptive_weighting" in config["config_training"].keys():
                self.lambda_bc = 1.0
                self.lambda_r = 1e-3

                self.freq_update_weightning = 25
                self.alpha_weighting = config["config_training"]["alpha_adaptive_weighting"]
                self.epsilon = 1e-4

        self.INN = INN
        self.config = config
        self.config_training = config["config_training"]

        # Save the conifguration file
        cleaned_config = remove_non_serializable(copy.deepcopy(self.config))
        self.save_hyperparameters(cleaned_config)

        #(Unnormalized) ground truth log likelihood
        if "log_p_target_name" in config["config_evaluation"].keys():
            log_p_target = log_p_target_dict[config["config_evaluation"]["log_p_target_name"]]
            self.log_p_target = partial(log_p_target,**config["config_evaluation"]["log_p_target_kwargs"])
        else:
            self.log_p_target = None

        #Log the best model
        self.best_mean_KL = None
        self.best_epoch_mean_KL = None

        ####################################################################################################################################
        # Initialize the data free model
        ####################################################################################################################################

        # Change the lengths of the different phases of beta sampling from epochs to iterations
        if "regularization_data_free" in self.config_training.keys():
            self.config["config_training"]["regularization_data_free_start"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["regularization_data_free_start"]
            self.config["config_training"]["regularization_data_free_full"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["regularization_data_free_full"]

        else:
            self.config["config_training"]["regularization_data_free_start"] = None

        if (self.config["config_training"]["regularization_data_free_start"] is not None) and ("t_burn_in" in self.config["config_training"]["loss_model_params"].keys()):
            self.config["config_training"]["loss_model_params"]["t_burn_in"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["loss_model_params"]["t_burn_in"]
            self.config["config_training"]["loss_model_params"]["t_full"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["loss_model_params"]["t_full"]

        self.iteration = 0

        N = self.config["config_data"]["N"]

        # Initialize the loss model
        factory = DataFreeLossFactory()
        self.data_free_loss_model = factory.create(
            key = self.config["config_training"]["data_free_loss_mode"],
            config=config
        )

        self.validation_data_loader_dict = {}

        #Load the validation data
        for k in np.arange(self.config["config_evaluation"]["kappa_validation_specs"][0],self.config["config_evaluation"]["kappa_validation_specs"][1],self.config["config_evaluation"]["kappa_validation_specs"][2]):
            for l in np.arange(self.config["config_evaluation"]["lambda_validation_specs"][0],self.config["config_evaluation"]["lambda_validation_specs"][1],self.config["config_evaluation"]["lambda_validation_specs"][2]):

                DS_i= DataSetScalarTheory2D_preprocessed_data(
                    N = N,
                    mode = "validation",
                    kappa_list = [round(k,4)],
                    lambda_list = [round(l,4)],
                    augment=True,
                    max_samples=self.config["config_evaluation"]["plain_samples_validation_set"],
                    sigma_noise=0.0
                )
                
                DL_i = DataLoader(
                    DS_i,
                    batch_size = self.config["config_evaluation"]["batch_size_validation"],
                    shuffle = True,
                    num_workers = 4
                )

                self.validation_data_loader_dict[f"k={round(k,4)}_l={round(l,4)}"] = DL_i

        # Load the reference data
        self.reference_simulation = np.loadtxt(f"./data/ScalarTheory/validation_data/N_{N}_LANGEVIN_SPECIFIC/summary_lambda_0.02_0.txt",skiprows=1)

        # Update counters if the training is a continuation of another training
        if "continue_training_kwargs" in config.keys():

            # Load the stored training progress
            reader_k = SummaryReader(config["continue_training_kwargs"]["base_experiment"])
            df_k = reader_k.scalars
            df_red = df_k[(df_k["tag"] == "epoch")]
            kl_k = df_red["value"].values

            last_epoch_base = max(kl_k)
            last_checkpoint_epoch = np.floor(last_epoch_base / config["config_evaluation"]["validation_freq"]) * config["config_evaluation"]["validation_freq"]

            print(f"last epoch in base run: {last_epoch_base}")
            print(f"last epcoh with checkpoint: {last_checkpoint_epoch}")

            self.iteration = last_checkpoint_epoch * config["config_training"]["n_batches_per_epoch"]

            if self.data_free_loss_model is not None:
                self.data_free_loss_model.iteration = last_checkpoint_epoch * config["config_training"]["n_batches_per_epoch"]

            # Initialize the adaptive weighting between the two loss cintributions
            if "alpha_adaptive_weighting" in config["config_training"].keys():

                df_red = df_k[(df_k["tag"] == "parameters/weighting_data_free")]
                kl_k = df_red["value"].values
                self.lambda_r = kl_k[-1]

    @property
    def regularization_data_free(self)->float:
        """
        Compute the weighting of the TS term in the total loss
        """

        #No regularization
        if not self.config["config_training"]["use_nll_loss"]:
            return 1.0

        #No regularization at the beginning
        if (self.data_free_loss_model is None) or (self.iteration < self.config_training["regularization_data_free_start"]):
            return 0.0
    
        #Full regularization after the ramp up phase
        elif self.iteration >= self.config_training["regularization_data_free_full"]:
            l = 1.0
        
        elif self.config_training["regularization_data_free_full"] == self.config_training["regularization_data_free_start"]:
            l = 1.0
        
        #Linear interpolation in between
        else:
            l = (self.iteration - self.config_training["regularization_data_free_start"]) / (self.config_training["regularization_data_free_full"] - self.config_training["regularization_data_free_start"])

        return l * self.config_training["regularization_data_free"]

    def configure_optimizers(self)->dict:
        """
        Initialize the optimizer and the learning rate scheduler
        """
        
        params = self.INN.parameters()

        # Configure the optimizer
        if self.config_training["optimizer_type"] != "LBFGS" and (self.config_training["optimizer_type"] in optimizer_dict.keys()):
            optimizer = optimizer_dict[self.config_training["optimizer_type"]](params = params, lr = self.config_training["lr"], weight_decay = self.config_training["weight_decay"])

        # LBFGS has reduced set of fearures. Has therefore to be treated differently
        elif self.config_training["optimizer_type"] == "LBFGS":
            optimizer = torch.optim.LBFGS(params = params, lr = self.config_training["lr"])
        
        else:
            raise NotImplementedError

        # Configure the learning rate scheduler
        if self.config["config_training"]["lr_scheduler_config"]["mode"]  == "exponential":
            
            gamma = self.config["config_training"]["lr_scheduler_config"]["final_lr_ratio"] ** (1 / self.config_training["n_epochs"])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = gamma)
            interval = "epoch"

        elif self.config["config_training"]["lr_scheduler_config"]["mode"]  == "multiCycle":
            
            epochs_per_cycle = self.config["config_training"]["lr_scheduler_config"]["epochs_per_cycle"]
            print("epochs per cycle: ",epochs_per_cycle)

            learning_rates = [self.config_training["lr"]]

            for i in range(len(self.config["config_training"]["lr_scheduler_config"]["lr_decay_factors"])):
                learning_rates.append(learning_rates[-1] * self.config["config_training"]["lr_scheduler_config"]["lr_decay_factors"][i])
            print("learning rates: ",learning_rates)

            n_cycles = len(epochs_per_cycle)

            scheduler = MultiCycleLR(
                optimizer=optimizer,
                epochs_per_cycle=epochs_per_cycle,
                n_cycles = n_cycles,
                steps_per_epoch=self.config_training["n_batches_per_epoch"],
                max_lrs = learning_rates
            )
            interval = "step"

        elif self.config["config_training"]["lr_scheduler_config"]["mode"]  == "oneCycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr = self.config_training["lr"],
                steps_per_epoch = self.config_training["n_batches_per_epoch"],
                epochs = self.config_training["n_epochs"],
            )
            interval = "step"

        else:
            raise ValueError("Unknown lr scheduler mode")
        
        return {"optimizer":optimizer,
                "lr_scheduler":{
                    "scheduler":scheduler,
                    "interval":interval
                    }
            }
        
    def state_dict(self)->dict:
        """
        Return the state dict of the invertible function
        """

        state_dict = {"INN":self.INN.inn.state_dict()}

        if self.config["config_model"]["process_beta_parameters"]["mode"] == "learnable":
            state_dict["Embedder"] = self.INN.beta_processing_function.state_dict()

        return state_dict

    def __compute_nll_objective(self,x_batch:torch.tensor,kappa_batch:torch.tensor)->torch.tensor:
        """
        Compute the negative log-likelihood objective.

        parameters:
            x_batch:        Batch of training data
            kappa_batch:    Batch of condition values corresponding to the training data

        returns:
            nll:            Negative log-likelihood objective
        """

        nll = - self.INN.log_prob(x_batch,kappa_batch).mean()

        return nll
    
    def __get_grad_magnitude(self,loss_i:torch.tensor)->float:
        """
        Compute the magnitude of the gradient of a loss with respect to the model parameters.

        parameters:
            loss_i:     Loss for which the gradient magnitude is computed

        returns:
            grad_mag:   Magnitude of the gradient with respect to the model parameters
        """

        #Compute the magnitude of the gradient for the nll loss
        opt = self.optimizers()
        opt.zero_grad()

        loss_i.backward()
    
        grad_mag = 0
        for name, param in self.INN.inn.named_parameters():
            if param.requires_grad and (param.grad is not None):
                grad_mag += param.grad.pow(2).sum().detach().item()
        grad_mag = np.sqrt(grad_mag)

        return grad_mag

    def __update_lambda_PINF(self,x_batch:torch.tensor,kappa_batch:torch.tensor,a:float)->None:
        """
        Update the parameters of the adaptive loss balancing scheme.

        parameters:
            x_batch:       Batch of training data
            kappa_batch:   Batch of condition values corresponding to the training data
            a:             Indicator if loss balancing is applied
        """

        if self.config["config_training"]["adaptive_weighting"] and (self.iteration % self.freq_update_weightning) == 0 and a > 0:
            
            # Compute the magnitude of the gradient for the nll loss
            eval_nll = self.__compute_nll_objective(x_batch=x_batch,kappa_batch=kappa_batch)
            mag_nll = self.__get_grad_magnitude(loss_i=eval_nll)

            eval_loss_data_free = self.data_free_loss_model(
                INN = self.INN,
                get_eval_points=self.get_evaluation_points,
                epoch = self.current_epoch,
                logger = self.logger
            )
            mag_PI = self.__get_grad_magnitude(loss_i=eval_loss_data_free)

            self.lambda_bc = self.alpha_weighting * self.lambda_bc + (1 - self.alpha_weighting) *  (mag_PI + mag_nll) / (mag_nll+self.epsilon)
            self.lambda_r = self.alpha_weighting * self.lambda_r + (1 - self.alpha_weighting) *  (mag_PI + mag_nll) / (mag_PI+self.epsilon)

    def training_step(self,batch:tuple,batch_idx:int)->dict:
        """
        Training step

        parameters:
            batch:      Batch of training data
            batch_idx:  Index of the batch

        returns:
            dictionary containing the loss
        """

        self.INN.train(True)

        # Log the learning rate
        self.log_dict({"parameters/lr":self.lr_schedulers().get_last_lr()[0]})

        loss = torch.zeros(1).to(self.device)

        # NLL loss
        if self.config["config_training"]["use_nll_loss"]:
            kappa_batch,lambda_batch,x_batch_plain = batch
            x_batch = self.transformation(x_batch_plain)
            assert(x_batch.shape == x_batch_plain.shape)

            nll = self.__compute_nll_objective(x_batch=x_batch,kappa_batch=kappa_batch)

            loss = loss + nll
            self.log_dict({"loss/nll":nll})

        # Data free loss contribution
        a = self.regularization_data_free
        if ((self.data_free_loss_model is not None) and (a > 0.0)) or not self.config["config_training"]["use_nll_loss"]:
            loss_data_free = self.data_free_loss_model(
                INN = self.INN,
                get_eval_points=self.get_evaluation_points,
                epoch = self.current_epoch,
                logger = self.logger
            )

            if self.config["config_training"]["adaptive_weighting"]:    
                a = self.fixed_relative_weighting * self.lambda_r / self.lambda_bc

            loss = loss + a *  loss_data_free

            self.log_dict({"loss/data_free":loss_data_free})
        
        self.log_dict({"parameters/weighting_data_free":a})

        # Update counter
        self.iteration += 1

        # Update internal counter of the data free model
        if self.data_free_loss_model is not None:
            self.data_free_loss_model.iteration += 1

        # Update the adaptive weighting of the different loss contributions
        if (a != 0) and self.config["config_training"]["use_nll_loss"]:
            self.__update_lambda_PINF(x_batch,kappa_batch,a)

        return {"loss":loss}
      
    def get_evaluation_points(self,beta_tensor:torch.Tensor)->torch.Tensor:
        """
        Sample points for the evaluation of the physics-informed loss contribution

        parameters:
            beta_tensor:    Condition values where an evaluation point should be sampled

        returns:
            x:              Batch of points following the model distribution at the specified condition values
        """

        with torch.no_grad():
            x = self.INN.sample(n_samples = len(beta_tensor),beta_tensor = beta_tensor)

            if ("symmetry_in_get_evaluation_points" in self.config["config_training"].keys()) and  self.config["config_training"]["symmetry_in_get_evaluation_points"]:
                x = self.transformation(x)

            if ("noise_level_in_get_evaluation" in self.config["config_training"].keys()):
                x = x + torch.randn_like(x) * self.config["config_training"]["noise_level_in_get_evaluation"]

        return x.detach()

    def validation(self)->None:
        """
        Evaluate the performance of the model.
        """

        if ((self.current_epoch + 1) % self.config["config_evaluation"]["validation_freq"] == 0) or (self.current_epoch == 0) or (self.current_epoch + 1 == self.config_training["n_epochs"]):

            KL_list = []
            ESS_r_list = []
            
            with torch.no_grad():
                self.INN.train(False)

                for k in np.arange(self.config["config_evaluation"]["kappa_validation_specs"][0],self.config["config_evaluation"]["kappa_validation_specs"][1],self.config["config_evaluation"]["kappa_validation_specs"][2]):
                    for l in np.arange(self.config["config_evaluation"]["lambda_validation_specs"][0],self.config["config_evaluation"]["lambda_validation_specs"][1],self.config["config_evaluation"]["lambda_validation_specs"][2]):

                        DL_i = self.validation_data_loader_dict[f"k={round(k,4)}_l={round(l,4)}"]

                        log_p_theta_val = torch.zeros([0])
                        log_p_theta_INN = torch.zeros([0])
                        log_p_target_INN = torch.zeros([0])

                        for j,(kappa_batch,lambda_batch,x_batch) in enumerate(DL_i):

                            # Compute the log likelihood of the validation set
                            log_p_theta_val_i = self.INN.log_prob(x_batch.to(self.config["device"]),beta_tensor=k)
                            log_p_theta_val = torch.cat((log_p_theta_val,log_p_theta_val_i.detach().cpu()),0)

                            # Compute the log likelihood of the INN samples
                            samples_i = self.INN.sample(n_samples = len(kappa_batch),beta_tensor = k)

                            log_p_theta_INN_i = self.INN.log_prob(samples_i,k).cpu()
                            log_p_theta_INN = torch.cat((log_p_theta_INN,log_p_theta_INN_i.detach().cpu()),0)

                            log_p_target_INN_i = - S_ScalarTheory(samples_i.cpu(),kappas = k,lambdas = l).cpu()
                            log_p_target_INN = torch.cat((log_p_target_INN,log_p_target_INN_i.detach().cpu()),0)

                        # Compuete the relative Kish effective sample size
                        log_omega = log_p_target_INN - log_p_theta_INN
                        log_a = 2 * torch.logsumexp(log_omega,0)
                        log_b = torch.logsumexp(2 * log_omega,0)

                        ESS_r = torch.exp(log_a - log_b) / len(log_omega)
                        ESS_r_list.append(ESS_r)

                        # Get the log likelihood of the validation set
                        nll_i = - log_p_theta_val.mean().item()
                        KL_list.append(nll_i)

            self.INN.train(True)

            if (self.best_mean_KL is None) or (self.best_mean_KL > np.mean(KL_list)):
                self.best_mean_KL = np.mean(KL_list)
                self.best_epoch_mean_KL = self.current_epoch
                
            self.current_mean_KL = np.mean(KL_list)
            self.current_mean_ESS_r = np.mean(ESS_r_list)

            self.log_dict({f"model_performance/mean_validation_KL":self.current_mean_KL,"model_performance/mean_validation_ESS_r":self.current_mean_ESS_r})

    def on_train_epoch_end(self)->None:
        """
        Evaluation and plotting
        """

        self.validation()

        if (((self.current_epoch + 1) % self.config["config_evaluation"]["plot_freq"] == 0) or (self.current_epoch + 1) == self.config_training["n_epochs"] or self.current_epoch == 0):

            
            self.INN.train(False)

            if self.config["config_evaluation"]["run_evaluations"]:

                ########################################################################################################################################################################################################
                # Evaluate physical observables
                ########################################################################################################################################################################################################

                kappas_phyiscs = np.arange(self.config["config_evaluation"]["kappa_physics_specs"][0],self.config["config_evaluation"]["kappa_physics_specs"][1],self.config["config_evaluation"]["kappa_physics_specs"][2])
                lambda_physics = np.arange(self.config["config_evaluation"]["lambda_physics_specs"][0],self.config["config_evaluation"]["lambda_physics_specs"][1],self.config["config_evaluation"]["lambda_physics_specs"][2])

                if len(lambda_physics) > 1:
                    raise NotImplementedError
                
                n_batches_phyiscs = int(self.config["config_evaluation"]["n_samples_physics"] / self.config["config_evaluation"]["batch_size_physics"])

                magnetizations = torch.zeros([len(kappas_phyiscs),2])
                actions_gt = torch.zeros([len(kappas_phyiscs),2])
                susceptibility = torch.zeros([len(kappas_phyiscs),2])
                binder_cumulant = torch.zeros([len(kappas_phyiscs),2])

                counter = 0
                
                for k in kappas_phyiscs:
                    for l in lambda_physics:

                        actions_gt_kl = torch.zeros([0])
                        magnetization_kl = torch.zeros([0])

                        for i in range(n_batches_phyiscs):
                            with torch.no_grad():
                                samples_kli = self.INN.sample(n_samples = self.config["config_evaluation"]["batch_size_physics"],beta_tensor = k).cpu().detach()

                                action_gt_kli = S_ScalarTheory(samples_kli,k,l)
                                magnetization_kli = samples_kli.sum(dim = (1,2,3))

                                actions_gt_kl = torch.cat((actions_gt_kl,action_gt_kli),0)
                                magnetization_kl = torch.cat((magnetization_kl,magnetization_kli),0)

                        N = self.config["config_data"]["N"]

                        mean_magnetization,std_magnetization = bootstrap(x = np.abs(np.array(magnetization_kl)) / N**2,s = np.mean,args={"axis":0})
                        mean_action_gt,std_action_gt = bootstrap(x = np.array(actions_gt_kl) / N**2,s = np.mean,args={"axis":0})
                        susceptibility_mean,sigma_susceptibility =bootstrap(x = np.abs(np.array(magnetization_kl)),s = get_susceptibility,args={"Omega":N**2})
                        U_L_mean,sigma_U_L = bootstrap(x = np.array(magnetization_kl),s = get_U_L,args={"Omega":N**2})

                        magnetizations[counter] = torch.Tensor([mean_magnetization,std_magnetization])
                        actions_gt[counter] = torch.Tensor([mean_action_gt,std_action_gt])
                        susceptibility[counter] = torch.Tensor([susceptibility_mean,sigma_susceptibility])
                        binder_cumulant[counter] = torch.Tensor([U_L_mean,sigma_U_L])

                        counter += 1

                properties = [
                    magnetizations,
                    actions_gt,
                    susceptibility,
                    binder_cumulant
                ]

                reference_properties = [
                    self.reference_simulation[:,1:3],
                    self.reference_simulation[:,3:5],
                    self.reference_simulation[:,7:9],
                    self.reference_simulation[:,5:7]
                ]

                y_lims = self.config["config_evaluation"]["y_lims"]

                labels = ["magnetization","action","susceptibility","binder_cumulant"]
                ylabels = ["|m|","s",r"$\chi^2$",r"$U_L$"]

                for i,prop in enumerate(properties):

                    fig,ax = plt.subplots(1,1,figsize = (10,5))

                    ax.errorbar(self.reference_simulation[:,0],reference_properties[i][:,0],yerr = reference_properties[i][:,1],ls = "",marker = "o",markersize = 3, capsize = 3, markeredgewidth = 1,color = "b",label = "MCMC")
                    ax.errorbar(kappas_phyiscs,prop[:,0],yerr = prop[:,1],ls = "",marker = "o",markersize = 3, capsize = 3, markeredgewidth = 1,color = "r",label = "INN")
                    ax.set_ylim(bottom = y_lims[i][0],top = y_lims[i][1])
                    ax.set_xlabel(r"$\kappa$")
                    ax.set_ylabel(ylabels[i])
                    ax.legend()

                    plt.tight_layout()
                    self.logger.experiment.add_figure(f'observables/{labels[i]}', fig, self.current_epoch + 1)
                    plt.close(fig)
            
            ########################################################################################################################################################################################################
            #Plot running average of the internal expectation values
            ########################################################################################################################################################################################################

            # TRADE grid loss
            if (self.config_training["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V2") and (self.regularization_data_free > 0.0):

                fig,ax = plt.subplots(1,1,figsize = (10,5))
                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),self.data_free_loss_model.EX_A.detach().cpu().numpy(),color = "r",ls = "",marker = "o")
                ax.plot(self.reference_EX_A[:,0],self.reference_EX_A[:,1],color = "b",ls = "dotted")
                ax.set_xlabel(r"$\kappa$")
                ax.set_ylabel(r"$\langle A \rangle$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/EX_A', fig, self.current_epoch + 1)
                plt.close(fig)

                fig,ax = plt.subplots(1,1,figsize = (10,5))

                loss_plot = self.data_free_loss_model.loss_statistics.detach().cpu()
                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),loss_plot,color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\kappa$")
                ax.set_ylabel(r"$\langle L(\kappa) \rangle$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/loss_per_bin', fig, self.current_epoch + 1)
                plt.close(fig)

                fig,ax = plt.subplots(1,1,figsize = (10,5))

                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),self.data_free_loss_model.log_causality_weights.detach().cpu(),color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\kappa$")
                ax.set_ylabel(r"$\log{\omega(\kappa)}$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/log_importance_weights', fig, self.current_epoch + 1)
                plt.close(fig)

                fig,ax = plt.subplots(1,1,figsize = (10,5))

                ax.plot(self.data_free_loss_model.param_grid.squeeze().cpu(),self.data_free_loss_model.log_causality_weights.detach().cpu().exp()/self.data_free_loss_model.log_causality_weights.detach().cpu().exp().sum(),color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\kappa$")
                ax.set_ylabel(r"$\omega(\kappa)$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/importance_weights', fig, self.current_epoch + 1)
                plt.close(fig)

                # Save the values in the dictionary where the treining progress is stored
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
            elif (self.config_training["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V2") and (self.regularization_data_free > 0.0):
                
                mask = (self.data_free_loss_model.param_storage_grid.squeeze().cpu() != 0)
                fig,ax = plt.subplots(1,1,figsize = (10,5))
                ax.plot(self.data_free_loss_model.param_storage_grid.squeeze().cpu()[mask],self.data_free_loss_model.EX_storage_grid.detach().cpu().numpy()[mask],color = "r",ls = "",marker = "o")
                ax.set_xlabel(r"$\kappa$")
                ax.set_ylabel(r"$\langle A \rangle$")

                plt.tight_layout()
                self.logger.experiment.add_figure(f'running_average/EX_A', fig, self.current_epoch + 1)
                plt.close(fig)

                # Save the values in the dictionary where the treining progress is stored
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
                
            ########################################################################################################################################################################################################
            # Plot the spin distribution
            ########################################################################################################################################################################################################

            with torch.no_grad():
                kappas_eval = np.arange(self.config["config_evaluation"]["kappa_validation_specs"][0],self.config["config_evaluation"]["kappa_validation_specs"][1],self.config["config_evaluation"]["kappa_validation_specs"][2])
                fig_spins,axes_spins = plt.subplots(len(kappas_eval),1,figsize = (10,5 * len(kappas_eval)))

                for counter,k in enumerate(kappas_eval):

                    spins_samples_kl = torch.zeros([0])
                    spins_val_kl = torch.zeros([0])

                    for l in np.arange(self.config["config_evaluation"]["lambda_validation_specs"][0],self.config["config_evaluation"]["lambda_validation_specs"][1],self.config["config_evaluation"]["lambda_validation_specs"][2]):

                        DL_i = self.validation_data_loader_dict[f"k={round(k,4)}_l={round(l,4)}"]

                        for j,(kappa_batch,lambda_batch,x_batch) in enumerate(DL_i):

                            samples_i = self.INN.sample(n_samples = len(kappa_batch),beta_tensor = k)

                            spins_samples_kl = torch.cat((spins_samples_kl,samples_i.flatten().cpu().detach()),0)
                            spins_val_kl = torch.cat((spins_val_kl,x_batch.flatten().cpu().detach()),0)

                    fs = 20
                    axes_spins[counter].hist(spins_samples_kl,bins = 50,alpha = 1.0,density = True,edgecolor = "r",lw = 2,label = "INN",histtype='step')
                    axes_spins[counter].hist(spins_val_kl,bins = 50,alpha = 1.0,density = True,edgecolor = "b",lw = 2,label = "MCMC",histtype='step')
                    axes_spins[counter].set_xlabel(r"$x_{i,j}$",fontsize = fs)
                    axes_spins[counter].set_ylabel(r"$p(x_{i,j})$",fontsize = fs)
                    axes_spins[counter].set_xlim([-6,6])
                    axes_spins[counter].set_title(r"$\kappa = $"+f"{round(k,4)}",fontsize = fs)
                    axes_spins[counter].legend(fontsize = fs)
                    axes_spins[counter].tick_params(axis='both', which='major', labelsize=fs)

                plt.tight_layout()
                self.logger.experiment.add_figure(f'spin_distribution', fig_spins, self.current_epoch + 1)
                plt.close(fig_spins)

            self.INN.train(True)
