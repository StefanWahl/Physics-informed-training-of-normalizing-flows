import torch
from torch.utils.data import Dataset
from typing import List
import os
import fnmatch
import json
import numpy as np

##################################################################################################
# 2D GMM power-scaling
##################################################################################################

class DataSet2DGMM(Dataset):
    def __init__(self,d:int,temperature_list:list[float],mode:str = "training",base_path:str = None,n_samples:int = 50000):
        
        if base_path is None:
            base_path = f"./data/2D_GMM/{mode}_data/"
        else:
            base_path = base_path + f"{mode}_data/"

        self.data = torch.zeros([0,d])
        self.beta = torch.zeros([0,1])

        for T in temperature_list:

            print(f"T = {T}:")

            folder_i = base_path+f"T_{T}_dim_{d}.pt"
            data_i = torch.load(folder_i)
            r=torch.randperm(len(data_i))
            data_i = data_i[r][:min([len(data_i),n_samples])]
            print(f"\t{len(data_i)} instances loaded")

            beta_tensor_i = torch.ones([len(data_i),1]) / T

            self.data = torch.cat((self.data,data_i),0)
            self.beta = torch.cat((self.beta,beta_tensor_i),0)

        print("Data set succesfully initialized\n\n")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index:int):
        return self.beta[index],self.data[index]

##################################################################################################
# 2D GMM with two external parameters
##################################################################################################

class DataSet_2D_ToyExample_external_two_parameters(Dataset):
    def __init__(self,d:int,parameter_coordinates:List[List[float]],mode:str = "training",base_path:str = None,n_samples:int = 50000):

        if base_path is None:
            base_path = f"./data/2D_Toy_two_external_parameters/{mode}_data/"
        else:
            base_path = base_path + f"{mode}_data/"

        self.data = torch.zeros([0,d])
        self.alpha_tensor = torch.zeros([0,1])
        self.beta_tensor = torch.zeros([0,1])

        for i in range(len(parameter_coordinates)):

            alpha_i = parameter_coordinates[i][0]
            beta_i = parameter_coordinates[i][1]

            print(f"alpha = {alpha_i}; beta = {beta_i}:")

            folder_i = os.path.join(base_path,f"alpha_{alpha_i}_beta_{beta_i}_dim_2.pt")
            data_i = torch.load(folder_i)

            r = torch.randperm(len(data_i))
            data_i = data_i[r][:min([len(data_i),n_samples])]
            print(f"\t{len(data_i)} instances loaded")

            alpha_tensor_i = alpha_i * torch.ones([len(data_i),1])
            beta_tensor_i = beta_i * torch.ones([len(data_i),1])

            self.data = torch.cat((self.data,data_i),0)
            self.alpha_tensor = torch.cat((self.alpha_tensor,alpha_tensor_i),0)
            self.beta_tensor = torch.cat((self.beta_tensor,beta_tensor_i),0)

        print("Data set succesfully initialized\n\n")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index:int):
        return self.alpha_tensor[index],self.beta_tensor[index],self.data[index]

##################################################################################################
# Scalar Theory
##################################################################################################

class DataSetScalarTheory2D_preprocessed_data(Dataset):
    def __init__(self,N:int,kappa_list:list[float],lambda_list:list[float],max_samples:int = 6250,mode:str = "training",augment:bool = False,sigma_noise:float = 0.0,base_path:str = None):
        '''
        parameters:
            N:                          Width and lenght of the lattice 
            kappa_list:                 List with used parameters kappa
            lambda_list:                List with used parameters lambda
            max_samples:                Number of samples loaded from the data set
            mode:                       Training or validation
            augment:                    Apply fixed set of transformations
            sigma_noise:                Standard deviation of the dequantization noise
            base_path:                  Location of the data set
        '''

        super().__init__()

        if base_path is None:
            base_path = f"./data/ScalarTheory/{mode}_data/"

        # Internal data storage
        self.data = torch.zeros([0,1,N,N])
        self.kappa_action = torch.zeros([0,1])
        self.lambda_action = torch.zeros([0,1])

        self.sigma_noise = sigma_noise

        print("Initialize dataset...")

        # Load the stored states
        for l in lambda_list:
            for k in kappa_list:

                print(f"kappa = {k}, lambda = {l}:")

                folder_i = base_path+f"N_{N}_LANGEVIN_SPECIFIC_Data_Set_curated/kappa_{k}_lambda_{l}/"
                data_i = torch.load(folder_i + "states_curated.pt")
                print(f"\t{len(data_i)} instances loaded")


                # Check consistency of the data
                # Get the content of the folder
                content_i = os.listdir(folder_i)
                
                # Get all the information files in the folder
                info_files = fnmatch.filter(content_i,"info_?.json")

                if len(info_files) == 0:
                    raise ValueError("No information files found in the folder. Abort.")
                
                for info_file in info_files:

                    with open(os.path.join(folder_i,info_file),"r") as file:
                        info_ij = json.load(file)
                    file.close()

                    assert(info_ij["N"] == N)
                    assert(info_ij["kappa_action"] == k)
                    assert(info_ij["lambda_action"] == l)
                    assert("processing_meta_data" in info_ij)

                    print(f"\t{info_file} is consistent")

                # Select the desired number of states
                indices_i = np.random.permutation(len(data_i))[:min([len(data_i),max_samples])]
                data_i = data_i[indices_i]

                # Use the symmetry of the problem to increase the number of training samples 
                if augment:
                    # Horizontal flip
                    data_horizontal_flip_i = torch.flip(data_i,[2])

                    # Vertical flip
                    data_vertical_flip_i = torch.flip(data_i,[3])

                    # Horizontal and vertical flip
                    data_horizontal_vertical_flip_i = torch.flip(data_i,[2,3])

                    data_i = torch.cat((data_i,data_horizontal_flip_i,data_vertical_flip_i,data_horizontal_vertical_flip_i),dim = 0)

                    # Use the negative data set
                    data_neg_i = -1 * data_i

                    data_i = torch.cat((data_i,data_neg_i),dim = 0)

                kappa_tensor_i = torch.ones([len(data_i),1]) * k
                lambda_tensor_i = torch.ones([len(data_i),1]) * l

                self.data = torch.cat((self.data,data_i),0)
                self.kappa_action = torch.cat((self.kappa_action,kappa_tensor_i),0)
                self.lambda_action = torch.cat((self.lambda_action,lambda_tensor_i),0)

                print(f"\t{len(data_i)} instances added to data set")
                print(f"\tsigma = {self.sigma_noise}\n")

        print("Data set succesfully initialized\n\n")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index:int):
        image = self.data[index] + torch.randn_like(self.data[index]) * self.sigma_noise
        return self.kappa_action[index],self.lambda_action[index],image
