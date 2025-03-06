import torch
from FrEIA.utils import force_to
import torch.distributions as D
import numpy as np

class GMM():
    def __init__(self,means:torch.tensor,covs:torch.tensor,weights:torch.tensor = None,device:str = "cpu")->None:
        """
        parameters:
            means:      Tensor of shape [M,d] containing the locations of the gaussian modes
            covs:       Tensor of shape [M,d,d] containing the covariance matrices of the gaussian modes
            weights:    Tensor of shape [M] containing the weights of the gaussian modes. Uniform weights are used if not specified
            devcie:     Device 
        """

        # Get dimensionality of the data set
        self.d = len(means[0])

        # Get the number of modes
        self.M = len(means)
        self.mode_list = []

        # Check weights
        if weights is None:
            self.weights = torch.ones(self.M) / self.M
        else:
            self.weights = weights

        if self.weights.sum() != 1.0: 
            raise ValueError()

        # Initialize the normal modes
        for i in range(self.M):
            self.mode_list.append(force_to(D.MultivariateNormal(loc = means[i],covariance_matrix = covs[i]),device))

    def __call__(self,x:torch.tensor)->torch.tensor:
        """
        Evaluate the pdf of the model.

        parameters:
            x: Tensor of shape [N,d] containing the evaluation points

        returns:
            p: Tensor of shape [N] contaiing the pdf value for the evaluation points
        """

        p =  self.log_prob(x).exp()
        return p
    
    def log_prob(self,x:torch.tensor)->torch.tensor:
        """
        Evaluate the log pdf of the model.

        parameters:
            x: Tensor of shape [N,d] containing the evaluation points

        returns:
            log_p: Tensor of shape [N] contaiing the log pdf value for the evaluation points
        """

        log_p_i_storage = torch.zeros([self.M,len(x)]).to(x.device)

        for i in range(self.M):
            log_p_i_storage[i] = self.mode_list[i].log_prob(x).squeeze()


        log_p = torch.logsumexp(log_p_i_storage + torch.log(self.weights).to(x.device)[:,None],dim = 0)

        return log_p
    
    def sample(self,N:int)->torch.tensor:
        """
        Generate samples following the distribution

        parameters:
            N: Number of samples

        return:
            s: Tensor of shape [N,d] containing the generated samples
        """
        weights = np.zeros(len(self.weights))
        weights[:-1] = self.weights[:-1].cpu().detach().numpy()
        weights[-1] = 1.0 - self.weights[:-1].cpu().detach().numpy().sum()
        i = np.random.choice(a = self.M,size = (N,),p = weights)
        u,c = np.unique(i,return_counts=True)

        s = torch.zeros([0,self.d])

        for i in range(self.M):
            s_i = self.mode_list[u[i]].sample([c[i]])
            s = torch.cat((s,s_i),dim = 0)

        return s
