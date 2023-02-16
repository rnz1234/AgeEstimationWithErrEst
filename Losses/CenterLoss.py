import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):

    def __init__(self,margin=1,criterion= None,reduction = None):
        super(CenterLoss,self).__init__()

        self.margin = 0
        self.reduction = reduction

    def forward(self, Embeddings, Centers,Labels,Mode='MSE'):

        CenterLoss = ((Embeddings - Centers(Labels))**2).sum(1)

        if self.reduction == 'mean':
            CenterLoss = CenterLoss.mean()

        if self.reduction == 'sum':
            CenterLoss = CenterLoss.sum()

        #CenterLoss = ((Embeddings - Centers(Labels)) ** 2).sum(1).mean()

        #CenterLoss = torch.sum((Embeddings - Centers(Labels)).abs(), 1)

        if Mode=='MSE':
            return CenterLoss


        Centers.weight.data = F.normalize(Centers.weight.data, p=2, dim=1, eps=1e-1)
        Dist = 1-torch.mm(Centers.weight.data,Centers.weight.data.transpose(0, 1))
        Dist += 1000000000 * torch.eye(n=Dist.shape[0], m=Dist.shape[1])

        Margin=0
        for i in range(Centers.num_embeddings):

            if (i-1)>=0:
                Dist1 = Dist[i,i-1].clone()
                Dist[i,i-1] += 1000000000

            if (i+1)<Centers.num_embeddings:
                Dist2 = Dist[i, i + 1].clone()
                Dist[i,i+1] += 1000000000

            NegIdx = torch.argmin(Dist[i,:])

            if (i-1)>=0:
                Margin += F.relu(Dist1-Dist[i,NegIdx] + Dist1)

            if (i+1)<Centers.num_embeddings:
                Margin += F.relu(Dist2-Dist[i,NegIdx] + Dist2)

        Margin /= Centers.num_embeddings

        return Margin+CenterLoss.mean()