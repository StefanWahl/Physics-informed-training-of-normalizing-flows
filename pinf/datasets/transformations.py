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
