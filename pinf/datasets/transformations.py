import torch
import numpy as np

class RandomChangeOfSign(torch.nn.Module):
    def __init__(self, p=0.5):
        super(RandomChangeOfSign,self).__init__()
        self.p = p

    def forward(self, img):
        # Do some transformations
        factor = (2*(torch.rand(1) < self.p).float() - 1).item()
        img_new = factor*img

        return img_new
    
class RandomHorizentalRoll(torch.nn.Module):
    def __init__(self):
        super(RandomHorizentalRoll,self).__init__()
        

    def forward(self, img):

        N = img.shape[-1]
        
        steps = torch.randint(low = int(-np.floor(N/2)),high = int(np.floor(N/2)),size = (1,)).item()

        img_new = torch.roll(img, shifts = steps, dims = -1)

        return img_new
    
class RandomVerticalRoll(torch.nn.Module):
    def __init__(self):
        super(RandomVerticalRoll,self).__init__()
        

    def forward(self, img):

        N = img.shape[-1]
        
        steps = torch.randint(low = int(-np.floor(N/2)),high = int(np.floor(N/2)),size = (1,)).item()

        img_new = torch.roll(img, shifts = steps, dims = -2)

        return img_new

class AddGaussianNoise(object):
    """
    Add Gaussian noise to the input tensor.
    """

    def __init__(self,std,mean = 0) -> None:
        """
        parameters:
            std:    Standard deviation of the Gaussian noise
            mean:   Mean of the Gaussian noise
        """

        self.mean = mean
        self.std = std

    def __call__(self,x):
        """
        Add Gaussian noise to the input tensor.

        parameters:
            x:  Input tensor
        
        return:
            y:  Input tensor with added Gaussian noise
        """
        
        r = self.mean + self.std * torch.randn_like(x)
        y = x + r      

        return y
    
    def __repr__(self) -> str:
        """
        String representation of the object.

        return:
            String representation of the object
        """

        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"