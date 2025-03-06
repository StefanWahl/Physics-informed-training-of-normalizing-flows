import torch
from torch.utils.data import Dataset

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