import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
from torch.nn import Parameter


class MetricLearningLoss(nn.Module):

    def __init__(self,margin=1,LabelsDistT=0,criterion= None,reduction = None):
        super(MetricLearningLoss,self).__init__()

        self.margin      = 1
        self.LabelsDistT = LabelsDistT
        self.reduction = reduction

    def forward(self, Embeddings, Labels,Mode = 'Random'):

        #the distance between all labels
        LabelsDist  = (Labels.unsqueeze(-1)-Labels.unsqueeze(0)).abs().float()
        LabelsDist += 1e5 * torch.eye(n=LabelsDist.shape[0], m=LabelsDist.shape[1])

        #find another sample with the same label
        PosIdx0     = torch.arange(Labels.shape[0]) #i coordinate
        PosIdx1     = LabelsDist.argmin(1)#j coordinate

        #use ONLY same-label samples as positives
        idx     = torch.where(LabelsDist[PosIdx0,PosIdx1] == 0)[0]
        PosIdx0 = PosIdx0[idx]
        PosIdx1 = PosIdx1[idx]



        # compute embedding distance
        FeaturesDist  = 1 - torch.mm(Embeddings, Embeddings.transpose(0, 1))

        PositiveFeaturesDist = FeaturesDist[PosIdx0, PosIdx1]

        # find negative samples idx
        if Mode == 'Random':
            NegIdx = torch.randint(high=Embeddings.shape[0], size = (idx.shape[0], 1))


        if Mode == 'Hard':

            #mark self-distance
            FeaturesDist += 1e5 * torch.eye(n=FeaturesDist.shape[0], m=FeaturesDist.shape[1])

            #avoid same-label samples as negatives
            FeaturesDist[LabelsDist <= self.LabelsDistT] = 1e5


            #find negative samples idx
            NegIdx = FeaturesDist.argmin(1)
            NegIdx = NegIdx[idx]

            #LabelsDist[PosIdx0,PosIdx1]
            #LabelsDist[PosIdx0, NegIdx.squeeze()].long()


        NegFeaturesDist = FeaturesDist[PosIdx0,NegIdx.squeeze()]
        Margin = F.relu(PositiveFeaturesDist-NegFeaturesDist +self.margin)

        if self.reduction == 'mean':
            Margin = Margin.mean()

        if self.reduction == 'sum':
            Margin = Margin.sum()

        return Margin
