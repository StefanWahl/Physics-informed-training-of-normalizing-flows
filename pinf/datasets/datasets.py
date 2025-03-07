import torch
from torch.utils.data import Dataset
from typing import List
import os

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
