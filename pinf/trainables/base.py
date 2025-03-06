import torch
import torchvision.transforms as Trafo
from functools import partial
import numpy as np
import lightning as L
import copy

from pinf.trainables.utils import (
    optimizer_dict,
    MultiCycleLR,
    remove_non_serializable
    )

from pinf.losses.factory import DataFreeLossFactory
from pinf.models.INN import INN_Model

class BaseTrainableObject_TemperatureScaling(L.LightningModule):
    def __init__(self,INN:INN_Model,config:dict)->None:
        """
        Base class for training of a confitional normalizing flow for power-scaling

        parameters:
            INN:    Normalizing flow to train
            config: Configuration file
        """

        super(BaseTrainableObject_TemperatureScaling, self).__init__()

        # Transformation for the data augmentation
        self.transformation = Trafo.Compose([])
        
        # Fixed ratio between the update strenght of the two loss contributions
        if "fixed_relative_weighting" in config["config_training"].keys():
            self.fixed_relative_weighting = config["config_training"]["fixed_relative_weighting"]

        if not config["config_training"]["use_nll_loss"]:
            config["config_training"]["adaptive_weighting"] = False

        # Adaptive weighting for different loss contributions
        if "alpha_adaptive_weighting" in config["config_training"].keys():
            self.lambda_bc = 1.0
            self.lambda_r = 1e-3

            self.freq_update_weightning = 25
            self.alpha_weighting = config["config_training"]["alpha_adaptive_weighting"]
            self.epsilon = 1e-4

        self.INN = INN
        self.config = config
        self.config_training = config["config_training"]

        self.base_betas = 1 / torch.tensor(config["config_data"]["init_data_set_params"]["temperature_list"])

        # Save the configuration file
        cleaned_config = remove_non_serializable(copy.deepcopy(self.config))
        self.save_hyperparameters(cleaned_config)

        # Log the best model
        self.best_mean_KL = None
        self.best_epoch_mean_KL = None

        ####################################################################################################################################
        #Initialize the data-free model
        ####################################################################################################################################

        #Change the lengths of the different phases of beta sampling from epochs to iterations
        if "regularization_data_free" in self.config_training.keys():
            self.config["config_training"]["regularization_data_free_start"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["regularization_data_free_start"]
            self.config["config_training"]["regularization_data_free_full"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["regularization_data_free_full"]

        else:
            self.config["config_training"]["regularization_data_free_start"] = None

        if (self.config["config_training"]["regularization_data_free_start"] is not None) and ("t_burn_in" in self.config["config_training"]["loss_model_params"].keys()):
            self.config["config_training"]["loss_model_params"]["t_burn_in"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["loss_model_params"]["t_burn_in"]
            self.config["config_training"]["loss_model_params"]["t_full"] = config["config_training"]["n_batches_per_epoch"] * self.config["config_training"]["loss_model_params"]["t_full"]

        if (config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param") or (config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V3") or (config["config_training"]["data_free_loss_mode"] == "PINF_parallel_Ground_Truth_one_param_V2") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V2") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param_V3") or (config["config_training"]["data_free_loss_mode"] == "PINF_local_Ground_Truth_one_param"):
            config["config_training"]["loss_model_params"]["base_params"] = self.base_betas
            config["config_training"]["loss_model_params"]["n_epochs"] = config["config_training"]["n_epochs"] - int(config["config_training"]["regularization_data_free_start"] / config["config_training"]["n_batches_per_epoch"])
            config["config_training"]["loss_model_params"]["n_batches_per_epoch"] = config["config_training"]["n_batches_per_epoch"]

        # Initialize the loss model
        factory = DataFreeLossFactory()

        self.data_free_loss_model = factory.create(
            key = self.config["config_training"]["data_free_loss_mode"],
            config=config
        )

        self.iteration = 0

    @property
    def regularization_data_free(self)->float:
        """
        Compute the weighting of the TS term in the total loss
        """

        # No regularization
        if not self.config["config_training"]["use_nll_loss"]:
            return 1.0

        # No regularization at the beginning
        if (self.data_free_loss_model is None) or (self.iteration < self.config_training["regularization_data_free_start"]):
            return 0.0
    
        # Full regularization after the ramp up phase
        elif self.iteration >= self.config_training["regularization_data_free_full"]:
            l = 1.0
        
        elif self.config_training["regularization_data_free_full"] == self.config_training["regularization_data_free_start"]:
            l = 1.0
        
        # Linear interpolation in between
        else:
            l = (self.iteration - self.config_training["regularization_data_free_start"]) / (self.config_training["regularization_data_free_full"] - self.config_training["regularization_data_free_start"])

        return l * self.config_training["regularization_data_free"]

    def configure_optimizers(self)->None:
        """
        Initialize the optimizer and the learning rate scheduler
        """

        params = self.INN.parameters()

        optimizer = optimizer_dict[self.config_training["optimizer_type"]](params = params, lr = self.config_training["lr"], weight_decay = self.config_training["weight_decay"])

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
        
    def state_dict(self):
        """
        Return the state dict of the invertible function
        """

        state_dict = {"INN":self.INN.inn.state_dict()}

        if self.config["config_model"]["process_beta_parameters"]["mode"] == "learnable":
            state_dict["Embedder"] = self.INN.beta_processing_function.state_dict()

        return state_dict

    def training_step(self,batch:tuple,batch_idx:int)->dict:
        """
        Training step

        parameters:
            batch:      Batch of training data
            batch_idx:  Index of the batch

        returns:
            dictionary containing the loss
        """

        # Log the learning rate
        self.log_dict({"parameters/lr":self.lr_schedulers().get_last_lr()[0]})

        loss = torch.zeros(1).to(self.device)

        # Reiwghting objective
        if ("use_reweighted_nll_loss" in self.config["config_training"]) and (self.config["config_training"]["use_reweighted_nll_loss"]):

            beta_min = self.config["config_training"]["loss_model_params"]["beta_min"]
            beta_max = self.config["config_training"]["loss_model_params"]["beta_max"]

            _,x_batch_plain = batch
            log_beta_tensor = (np.log(beta_max) - np.log(beta_min)) * torch.rand([len(x_batch_plain),1]).to(self.device) + np.log(beta_min)
            beta_batch = log_beta_tensor.exp()
        
            x_batch = self.transformation(x_batch_plain)

            assert(x_batch.shape == x_batch_plain.shape)
            assert(len(self.base_betas) == 1)

            # Get the energy at the base temperature
            e_base = self.S(x_batch,1.0,**self.S_kwargs).squeeze()
            min_nll = 0.0

            log_w_T = (- e_base + min_nll + 2) * (beta_batch.squeeze() -  self.base_betas[0].item())

            nll_at_T = - self.INN.log_prob(x_batch,beta_batch)

            assert(nll_at_T.shape == log_w_T.shape)

            loss = (nll_at_T * log_w_T.exp()).mean()

            self.log_dict({"loss/reweighted_nll":loss})

            return {"loss":loss}

        # NLL loss
        if self.config["config_training"]["use_nll_loss"]:
            beta_batch,x_batch_plain = batch
            x_batch = self.transformation(x_batch_plain)

            assert(x_batch.shape == x_batch_plain.shape)

            nll = self.__compute_nll_objective(x_batch=x_batch,beta_batch=beta_batch)

            loss = loss + nll
            self.log_dict({"loss/nll":nll})

        # Data-free loss contribution
        a = self.regularization_data_free

        if ((self.data_free_loss_model is not None) and (a > 0.0)) or not self.config["config_training"]["use_nll_loss"]:
            loss_data_free = self.data_free_loss_model(
                INN = self.INN,
                get_eval_points=self.get_evaluation_points,
                epoch = self.current_epoch,
                logger = self.logger
            )

            # Adaptive loss balancing
            if self.config["config_training"]["adaptive_weighting"]:    
                a = self.fixed_relative_weighting * self.lambda_r / self.lambda_bc

            loss = loss + a *  loss_data_free

            self.log_dict({"loss/data_free":loss_data_free})
        
        self.log_dict({"parameters/weighting_data_free":a,"loss/total_loss":loss})

        # Update counter
        self.iteration += 1

        # Update internal counter of the data free model
        if self.data_free_loss_model is not None:
            self.data_free_loss_model.iteration += 1

        # Update the parameters of the adaptive loss balancing
        if self.config["config_training"]["use_nll_loss"] and (self.data_free_loss_model is not None):
            self.__update_lambda_PINF(x_batch,beta_batch,a)

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

        return x.detach()

    def on_train_epoch_end(self):
        pass

    def validation(self):
        pass
    
    def __compute_nll_objective(self,x_batch:torch.tensor,beta_batch:torch.tensor):
        """
        Compute the negative log-likelihood objective.

        parameters:
            x_batch:    Batch of training data
            beta_batch: Batch of condition values corresponding to the training data

        returns:
            nll:        Negative log-likelihood objective
        """

        nll = - self.INN.log_prob(x_batch,beta_batch).mean()

        return nll
    
    def __get_grad_magnitude(self,loss_i:torch.tensor):
        """
        Compute the magnitude of the gradient of a loss with respect to the model parameters.

        parameters:
            loss_i:     Loss for which the gradient magnitude is computed

        returns:
            grad_mag:   Magnitude of the gradient with respect to the model parameters
        """

        opt = self.optimizers()
        opt.zero_grad()

        loss_i.backward()
    
        grad_mag = 0

        for name, param in self.INN.inn.named_parameters():
            if param.requires_grad and (param.grad is not None):
                grad_mag += param.grad.pow(2).sum().detach().item()
        grad_mag = np.sqrt(grad_mag)

        return grad_mag

    def __update_lambda_PINF(self,x_batch:torch.tensor,beta_batch:torch.tensor,a:float)->None:
        """
        Update the parameters of the adaptive loss balancing scheme.

        parameters:
            x_batch:       Batch of training data
            beta_batch:    Batch of condition values corresponding to the training data
            a:             Indicator if loss balancing is applied
        """

        if self.config["config_training"]["adaptive_weighting"] and (self.iteration % self.freq_update_weightning) == 0 and a > 0:
            
            #Compute the magnitude of the gradient for the nll loss
            eval_nll = self.__compute_nll_objective(x_batch=x_batch,beta_batch=beta_batch)
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
