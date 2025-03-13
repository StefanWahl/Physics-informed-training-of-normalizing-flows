import torch
import lightning as L
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader

from pinf.datasets.datasets import get_EMNIST_datasets
from pinf.models.INN import INN_Model
from pinf.trainables.utils import (
    optimizer_dict,
    remove_non_serializable
    )

class TrainingObject_EMNIST(L.LightningModule):
    """
    Train a normalizing flow on MNIST-like data
    """
    def __init__(self,INN:INN_Model,config:dict)->None:
        """
        parameters:
            INN:    Normalizing flow to train
            config: Configuration file
        """

        super(TrainingObject_EMNIST,self).__init__()

        self.INN = INN
        self.config = config

        # Save the conifguration file
        cleaned_config = remove_non_serializable(copy.deepcopy(self.config))
        self.save_hyperparameters(cleaned_config)

        # Get the validation data loader
        _,DS_validation = get_EMNIST_datasets(
            split="digits",
            data_folder = config["config_data"]["data_root_folder"],
            **config["config_data"]["init_data_set_params"]
        )

        DL_validation = DataLoader(
            DS_validation,
            batch_size = config["config_evaluation"]["bs_evaluation"],
            shuffle = False
            )
    
        self.validation_data_loader = DL_validation

    def training_step(self,batch:tuple,batch_idx:int)->dict:
        """
        Training step.

        parameters:
            batch:      Batch of data
            batch_idx:  Index of the batch
        
        returns:
            Dictionary containing the loss
        """

        self.INN.train(True)

        # Log the learning rate
        self.log_dict({"parameters/lr":self.lr_schedulers().get_last_lr()[0]})

        (x,y) = batch

        # Compute the negative log-likelihood objective
        nll = -self.INN.log_prob(x = x,beta_tensor = y.reshape(-1,1)).mean()

        self.log_dict({"loss/nll":nll})

        return {"loss":nll}

    def configure_optimizers(self)->dict:
        """
        Initialize the optimizer and the learning rate scheduler
        """
        
        params = self.INN.parameters()

        optimizer = optimizer_dict[self.config["config_training"]["optimizer_type"]](params = params, lr = self.config["config_training"]["lr"], weight_decay = self.config["config_training"]["weight_decay"])

        if self.config["config_training"]["lr_scheduler_config"]["mode"]  == "oneCycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr = self.config["config_training"]["lr"],
                steps_per_epoch = self.config["config_training"]["n_batches_per_epoch"],
                epochs = self.config["config_training"]["n_epochs"],
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
        Return the statedict of the invertible function
        """

        state_dict = {"INN":self.INN.inn.state_dict()}

        return state_dict
        
    def on_train_epoch_end(self)->None:
        self.validation()

        if (((self.current_epoch + 1) % self.config["config_evaluation"]["plot_freq"] == 0) or (self.current_epoch + 1) == self.config["config_training"]["n_epochs"] or self.current_epoch == 0) and self.config["config_evaluation"]["run_evaluations"]:

            # Plot INN samples
            with torch.no_grad():

                self.INN.eval()

                # Get conditions
                beta_tensor = torch.ones(self.config["config_data"]["n_classes"],self.config["config_evaluation"]["n_plot_samples_per_class"],device = self.config["device"])
                beta_tensor *= torch.arange(self.config["config_data"]["n_classes"],device = self.config["device"]).reshape(-1,1)
                beta_tensor = beta_tensor.reshape(-1,1).long()

                x_INN = self.INN.sample(n_samples = len(beta_tensor),beta_tensor = beta_tensor).detach().cpu()

                n_channels = x_INN.shape[1]
                x_INN = x_INN.squeeze()

                fig,ax = plt.subplots(self.config["config_data"]["n_classes"],self.config["config_evaluation"]["n_plot_samples_per_class"],figsize = (self.config["config_evaluation"]["n_plot_samples_per_class"] * 5,self.config["config_data"]["n_classes"] * 5))

                for i in range(self.config["config_data"]["n_classes"]):
                    for j in range(self.config["config_evaluation"]["n_plot_samples_per_class"]):
                        
                        ax[i][j].imshow(x_INN[i * self.config["config_evaluation"]["n_plot_samples_per_class"] + j],cmap = "gray")
                        ax[i][j].axis("off")


                plt.tight_layout()
                self.logger.experiment.add_figure('INN_samples', fig, self.current_epoch + 1)
                plt.close(fig)

                self.INN.train(True)

            # Validation samples
            plot_samples = [torch.zeros([0,n_channels,x_INN.shape[-2],x_INN.shape[-2]]) for i in range(self.config["config_data"]["n_classes"])]
            val_iterator = iter(self.validation_data_loader)

            while True:
                (x_val_i,y_val_i) = next(val_iterator)

                flag = True

                for i in range(self.config["config_data"]["n_classes"]):

                    if len(plot_samples[i]) == self.config["config_evaluation"]["n_plot_samples_per_class"]:
                        continue
                    
                    else: 
                        flag = False


                    mask_i = (y_val_i == i)

                    plot_samples[i] = torch.cat((plot_samples[i],x_val_i[mask_i]))
                    plot_samples[i] = plot_samples[i][:min(len(plot_samples[i]),self.config["config_evaluation"]["n_plot_samples_per_class"])]

                if flag:
                    break

            fig,ax = plt.subplots(self.config["config_data"]["n_classes"],self.config["config_evaluation"]["n_plot_samples_per_class"],figsize = (self.config["config_evaluation"]["n_plot_samples_per_class"] * 5,self.config["config_data"]["n_classes"] * 5))

            for i in range(self.config["config_data"]["n_classes"]):
                for j in range(self.config["config_evaluation"]["n_plot_samples_per_class"]):
                    
                    ax[i][j].imshow(plot_samples[i][j].squeeze(),cmap = "gray")
                    ax[i][j].axis("off")


            plt.tight_layout()
            self.logger.experiment.add_figure('validation_samples', fig, self.current_epoch + 1)
            plt.close(fig)

    def validation(self)->None:
        """
        Compute the average validation negative log-likelihood.
        """

        if ((self.current_epoch + 1) % self.config["config_evaluation"]["validation_freq"] == 0) or (self.current_epoch == 0) or (self.current_epoch + 1 == self.config["config_training"]["n_epochs"]):

            self.INN.eval()

            with torch.no_grad():

                validation_nll = 0.0

                for (x,y) in self.validation_data_loader:

                    nll = -self.INN.log_prob(x = x.to(self.config["device"]),beta_tensor = y.reshape(-1,1).to(self.config["device"])).sum()

                    validation_nll += nll

                validation_nll /= len(self.validation_data_loader.dataset)

                self.log_dict({"model_performance/mean_validation_nll":validation_nll})

                self.INN.train(True)
