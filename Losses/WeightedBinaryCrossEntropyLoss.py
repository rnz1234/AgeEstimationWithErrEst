import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class WeightedBinaryCrossEntropyLoss(nn.Module):

	def __init__(self, num_labels, device, reduction='None', binary_loss_type='BCE'):
		super(WeightedBinaryCrossEntropyLoss, self).__init__()

		self.reduction = reduction

		num_labels = int(num_labels)
		self.OneHotLabels = torch.ones((num_labels, num_labels), dtype=torch.float)
		self.OneHotLabels = torch.tril(self.OneHotLabels)
		self.OneHotLabels = torch.cat((torch.zeros((1, num_labels), dtype=torch.float), self.OneHotLabels), 0).to(device)

		# self.Alpha = torch.ones((NumLabes,NumLabes),dtype=torch.float)

		self.BinaryLossType = binary_loss_type.lower()
		self.Criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

		if binary_loss_type.lower() == 'focal':
			self.alpha = 0.5
			self.gamma = 2.0

	def forward(self, input, target, compute_weights=False, weights=None):

		result = self.Criterion(input, self.OneHotLabels[target, :])

		if self.BinaryLossType == 'focal':
			pt = torch.exp(-result)
			result = self.alpha * (1 - pt) ** self.gamma * result

		weights = None
		if compute_weights:
			EstimatedResult = (input > 0).sum(1)
			Error = (EstimatedResult - target).abs()
			weights = Error + 1
			weights[weights > 7] = 7
		# weights[weights<2]  = 0

		if weights is not None:
			result = weights.view(-1, 1) * result

		if self.reduction == 'mean':
			result = result.mean()

		if self.reduction == 'sum':
			result = result.sum()

		return result
