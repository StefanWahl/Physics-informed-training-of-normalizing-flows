import argparse
import json
from datetime import date
import torch
from torch.utils.data import DataLoader
import lightning as L
import os
from functools import partial
import numpy as np
import random
import yaml
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger 

from pinf.models.construct_INN_2D_GMM import set_up_sequence_INN_2D_GMM
from pinf.models.construct_INN_2D_GMM_two_parameters import set_up_sequence_INN_2D_ToyExample_two_parameters
from pinf.models.construct_INN_ScalarTheory import set_up_sequence_INN_ScalarTheory
from pinf.models.construct_INN_EMNIST import set_up_sequence_INN_MNIST_like

from pinf.datasets.datasets import (
    DataSet2DGMM,
    DataSet_2D_ToyExample_external_two_parameters,
    DataSetScalarTheory2D_preprocessed_data,
    get_EMNIST_datasets
    )

from pinf.trainables.GMM import TrainingObject_2D_GMM
from pinf.trainables.GMM_two_params import TrainingObject_2D_ToyExample_two_external_parameters
from pinf.trainables.ScalarTheory import TrainingObject_2D_Scalar_Theory
from pinf.trainables.EMNIST_digits import TrainingObject_EMNIST

data_set_class_dict = {
    "2D_GMM":DataSet2DGMM,
    "2D_ToyExample_two_external_parameters":DataSet_2D_ToyExample_external_two_parameters,
    "DataSetScalarTheory2D_preprocessed_data":DataSetScalarTheory2D_preprocessed_data,
    "EMNIST_digits":partial(get_EMNIST_datasets,training_data_only = True, split = "digits",data_folder = "./data/")
}

INN_constructor_dict = {
    "set_up_sequence_INN_2D_GMM":set_up_sequence_INN_2D_GMM,
    "set_up_sequence_INN_2D_ToyExample_two_parameters":set_up_sequence_INN_2D_ToyExample_two_parameters,
    "set_up_sequence_INN_ScalarTheory":set_up_sequence_INN_ScalarTheory,
    "set_up_sequence_INN_MNIST_like":set_up_sequence_INN_MNIST_like
}

trainable_dict = {
    "2D_GMM":TrainingObject_2D_GMM,
    "2D_ToyExample_two_external_parameters":TrainingObject_2D_ToyExample_two_external_parameters,
    "ScalarTheory":TrainingObject_2D_Scalar_Theory,
    "EMNIST_digits":TrainingObject_EMNIST
}

def get_configuration(args):
    """
    Load the configuration file
    """

    # Use the provided path
    if args.config_path is not None:
        file_path = args.config_path
        config_folder = os.path.dirname(file_path)

    # Use pretrained model
    elif args.experiment_to_continue is not None:
        print("Load training and continue...")

        # Load the configuration file
        config = yaml.safe_load(Path(args.experiment_to_continue + "/hparams.yaml").read_text())
        return config
    
    else:
        raise NotImplementedError()

    # Load the configuration
    with open(file_path,"r") as f:
        config = json.load(f)
    f.close()
    
    # Load the specified sub-configurations
    for key in config["sub_config_files"].keys():

        with open(os.path.join(config_folder,config["sub_config_files"][key]),"r") as f:
            config[key] = json.load(f)
        f.close()
    
    return config

def runner(config:dict,callbacks:list = [],load_INN_path:str = None)->L.LightningModule:
    """
    Train the model

    parameters:
        config:             Training configuration
        callbacks:          Training callbacks
        load_INN_path:      Path to a previous experiment to continue

    returns:
        training_object:    The used training object

    """

    # Get the data set
    if "data_set_class_name_training_data" in config["config_data"].keys():
        DS_training = data_set_class_dict[config["config_data"]["data_set_class_name_training_data"]](**config["config_data"]["init_data_set_params"])

    else:
        DS_training = data_set_class_dict[config["config_data"]["data_set_name"]](**config["config_data"]["init_data_set_params"])

    DL_training = DataLoader(
        DS_training, 
        batch_size=config["config_training"]["batch_size_nll"], 
        shuffle=True,
        num_workers=11,
        )
    
    # Get the number of training batches
    config["config_training"]["n_batches_per_epoch"] = len(DL_training)

    # Get the INN
    INN = INN_constructor_dict[config["config_model"]["set_up_function_name"]](config,DS_training)

    # Load pretrained INN
    if load_INN_path is not None:
        print(f"Load pretrained INN {load_INN_path}")
        INN.load_state_dict(path = load_INN_path)

    # Initialize the training object
    training_object = trainable_dict[config["config_data"]["data_set_name"]](INN = INN,config = config)

    # Set the device for the trainer
    if config["device"] == "cpu":
        accelerator = "cpu"
    elif config["device"] == "cuda:0":
        accelerator = "gpu"
    else:
        raise ValueError("Unkown device")
    
    # Initialize the logger
    logger = TensorBoardLogger(
        save_dir = config["logging_path"]
    )

    # Initialize the trainer
    trainer = L.Trainer(
        logger = logger,
        max_epochs = config["config_training"]["n_epochs"],
        default_root_dir = config["logging_path"],
        callbacks = callbacks,
        gradient_clip_val = config["config_training"]["gradient_clip_val"],
        log_every_n_steps = config["config_evaluation"]["log_scalars_freq"],
        enable_checkpointing=config["config_training"]["enable_checkpointing"],
        accelerator=accelerator
        )

    # Load the training progress if the training is initialized from a checkpoint
    if "continue_training_kwargs" in config.keys():
        state_dict = torch.load(config["continue_training_kwargs"]["loaded_model"])
        state_dict["state_dict"] = {}
        torch.save(state_dict,os.path.join(config["continue_training_kwargs"]["loaded_model"].rpartition('/')[0],"reduced.ckpt"))

        # Fit the model
        trainer.fit(
            training_object,
            DL_training,
            ckpt_path = os.path.join(config["continue_training_kwargs"]["loaded_model"].rpartition('/')[0],"reduced.ckpt")
            )
        
    else:
        # Fit the model
        trainer.fit(
            training_object,
            DL_training
            )
    
    return training_object

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',                type = str, default = None,         help="Location of the configuration file.")
    parser.add_argument('--tag',                        type = str, default = "debug",      help="Tag for the training run.")
    parser.add_argument('--experiment_to_continue',     type = str, default = None,         help="Location of an experiment to continue")

    args = parser.parse_args()

    # Get the configuration
    config = get_configuration(
        args = args
    )

    # Set the logging path
    if config["config_data"]["data_set_name"] == "2D_GMM":
        day_date = date.today()
        logging_path = f'./results/runs_2D_GMM/{day_date}_{args.tag}/'
    
    elif config["config_data"]["data_set_name"] == "2D_ToyExample_two_external_parameters":
        day_date = date.today()
        logging_path = f"./results/runs_2D_ToyExample_two_external_parameters/{day_date}_{args.tag}/"
    
    elif config["config_data"]["data_set_name"] == "ScalarTheory":
        day_date = date.today()
        logging_path = f"./results/runs_ScalarTheory/{day_date}_{args.tag}_N{config['config_data']['N']}/"

    elif config["config_data"]["data_set_name"] == "EMNIST_digits":
        day_date = date.today()
        logging_path = f"./results/runs_EMNIST_digits/{day_date}_{args.tag}/"

    else:
        raise ValueError("Data set not supported.")

    # Overwrite the logging path in case of continued training
    if args.experiment_to_continue is not None:
        logging_bath_base = config.get("logging_path").split("/")[-2]

        # Get the version number
        version = args.experiment_to_continue.split("/")[-1]

        logging_path = f"./results/runs_{config['config_data'].get('data_set_name')}/continued_{logging_bath_base}_{version}/"

        config["continue_training_kwargs"]={
            "base_experiment":args.experiment_to_continue,
            "loaded_model":os.path.join(args.experiment_to_continue,'checkpoints/last.ckpt')
        }
    
    # Set the random seed
    torch.manual_seed(seed = config["config_training"]["random_seed"])
    np.random.seed(config["config_training"]["random_seed"])
    random.seed(config["config_training"]["random_seed"])
    
    # Set the logging path
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config["device"] = device
    config["logging_path"] = logging_path

    # Checkpointing callback
    callbacks = []

    if config["config_training"]["enable_checkpointing"]:
        model_checkpoint = ModelCheckpoint(
            monitor='model_performance/mean_validation_KL',
            mode='min',
            save_top_k=1,
            filename='checkpoint_{epoch:02d}',
            verbose=True,
            save_last = True,
            every_n_epochs = config["config_evaluation"]["validation_freq"]
        )
        callbacks.append(model_checkpoint)

    # Set up and run the training
    if args.experiment_to_continue is not None:
        runner(config,callbacks = callbacks,load_INN_path = config["continue_training_kwargs"]["loaded_model"])

    else:
        runner(config,callbacks = callbacks,load_INN_path = None)
