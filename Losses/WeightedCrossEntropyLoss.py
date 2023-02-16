import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

        self.CE_Loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):

        TestResults = input.argmax(1)

        error = (TestResults - target).abs().float()

        Weights = error
        idx     = error>3
        Weights = Weights[idx]**2
        Weights /= Weights.sum()

        losses = self.CE_Loss(input, target)
        losses = losses[error>3]
        losses *= Weights

        Results = {}
        Results['loss']        = losses.sum()
        Results['NumElements'] = losses.shape[0]

        return Results