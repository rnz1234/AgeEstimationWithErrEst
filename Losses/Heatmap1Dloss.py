import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter1d


class Heatmap1Dloss(nn.Module):

    def __init__(self,device,NumLabes,sigma,reduction = 'mean'):
        super(Heatmap1Dloss, self).__init__()

        self.reduction = reduction

        NumLabes = int(NumLabes)
        self.GaussianVector = np.eye(NumLabes,dtype=np.float)
        self.GaussianVector = torch.Tensor(gaussian_filter1d(self.GaussianVector,axis=1, sigma=sigma))
        #plot(self.GaussianVector[5, :].cpu());show()

    def forward(self, input, target):

        Result = ((input - self.GaussianVector[target,:])**2).sum(1)

        if self.reduction == 'mean':
            Result = Result.mean()

        return Result