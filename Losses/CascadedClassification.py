import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
from torch.nn import Parameter

from Losses.AngularPenaltySMLoss import AngularPenaltySMLoss
from Losses.MeanVarianceLoss import keller_MeanVarianceLoss
from Losses.MetricLearningLoss import MetricLearningLoss
from Losses.WeightedBinaryCrossEntropyLoss import WeightedBinaryCrossEntropyLoss


class CascadedClassification(nn.Module):

	def __init__(self, NumClasses, CasscadeSize, CascadeSkip):
		super(CascadedClassification, self).__init__()

		self.NumClasses = int(NumClasses)
		self.CasscadeSize = int(CasscadeSize)

		self.CascadeSkip = int(CascadeSkip)

		self.ClassCenters = range(0, self.NumClasses * self.CascadeSkip, self.CascadeSkip)

		# Losses
		self.criterion = nn.CrossEntropyLoss()

		# self.CascadeLoss = nn.CrossEntropyLoss(weight=WeightsPerLabel)
		self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
		self.CenterLoss = nn.MSELoss(reduction='mean')
		self.MeanVarLoss = keller_MeanVarianceLoss(LamdaMean=0.2, LamdaSTD=0.05)
		self.BceLoss = WeightedBinaryCrossEntropyLoss(self.CasscadeSize, reduction='mean')
		self.MetricLearnLoss = MetricLearningLoss(LabelsDistT=3, reduction='mean')
		# self.AngularleLoss = AngleLoss()
		self.AngularleLoss = AngularPenaltySMLoss(256, self.CasscadeSize,
		                                          loss_type='arcface')  # loss_type in ['arcface', 'sphereface', 'cosface']

	def forward(self, Mode, input, QuantizedLabels, Labels, UseLoss, EmbeddingCenters=[], ApplyAngularLoss=[]):

		self.AngularleLoss = self.AngularleLoss.to(input['Class'].device)

		# greedy loss
		ClassLosses = self.criterion(input['Class'], QuantizedLabels)

		# ClassResult = input['Class'].argmax(1)
		# ClassError  = (ClassResult-Labels).abs()
		# ClasAccuracy = (input['ClassResult']==QuantizedLabels).float().mean()

		# BaseOrdinalResult = (input['OrdinalClass'] > 0).sum(1)

		if Mode == 'Cascade':

			# greedy
			GreedyLoss = 0
			CenterLoss = 0
			ProbLoss = 0
			CascadedClassLoss = 0
			SphereLoss = 0
			MVloss = 0
			OrdinalLoss = 0
			MlLoss = 0
			RegressionLoss = 0

			NoSamples = 0
			NoSamplesSphere = 0

			HalfCascade = int(self.CasscadeSize / 2)

			# loop over range classifers
			for i in range(len(self.ClassCenters)):

				Start = self.ClassCenters[i] - HalfCascade
				End = Start + self.CasscadeSize - 1

				CurrentsIdx = torch.where((Labels >= Start) & (Labels <= End))[0]
				# CurrentsIdx = torch.where(Labels == i)[0]

				# CurrentsIdx = torch.where((BaseOrdinalResult >= Start) & (BaseOrdinalResult <= End))[0]
				# CurrentsIdx = torch.where(BaseOrdinalResult == i)[0]

				# choose classifcation results in the range
				# CurrentsIdx = torch.where((ClassResult >= Start) & (ClassResult <= End) )[0]
				# CurrentsIdx = torch.where(ClassResult == i)[0]

				if CurrentsIdx.shape[0] == 0: continue

				CurrentLabels = Labels[CurrentsIdx] - Start
				# CurrentLabels   = torch.clamp(CurrentLabels,0,self.CasscadeSize-1)

				if UseLoss['Class']:
					CascadedClassLoss += self.CrossEntropyLoss(input['CascadeClassScores'][i][CurrentsIdx, :],
					                                           CurrentLabels)

				if UseLoss['Ordinal']:
					OrdinalLoss += self.BceLoss(input['CascadeOrdinalEmbed'][i][CurrentsIdx, :], CurrentLabels,
					                            ComputeWeights=False)
				# OrdinalLoss += (((input['CascadeOrdinalEmbed'][i][CurrentsIdx, :] > 0).sum(1) - CurrentLabels.float()) ** 2).mean()

				if UseLoss['Sphere']:
					# SphereLoss      += self.AngularleLoss( (input['CascadeSphere'][i][0][CurrentsIdx,:],input['CascadeSphere'][i][1][CurrentsIdx,:]), CurrentLabels)
					SphereLoss = self.AngularleLoss(input['CascadeEmbed'][i][CurrentsIdx, :], CurrentLabels)

				if UseLoss['Regress']:
					# the labels are around the center
					RegressionLoss += ((input['CascadeRegress'][i][CurrentsIdx].squeeze() - (
								CurrentLabels.squeeze().float() - HalfCascade)) ** 2).mean()
				# aa=(input['CascadeRegress'][i][CurrentsIdx].squeeze()*HalfCascade- (CurrentLabels.squeeze().float()-HalfCascade)).shape

				# FUll cascade log-prob loss
				# GreedyLoss += self.criterion(input['FullCascadeClassScores'][:, :, i], Labels)

				if UseLoss['MeanVar']:
					MVloss += self.MeanVarLoss(input['CascadeClassScores'][i][CurrentsIdx, :], CurrentLabels)

				if UseLoss['MetricLearning']:
					if CurrentsIdx.shape[0] > 10:
						MlLoss += self.MetricLearnLoss(input['CascadeEmbed'][i][CurrentsIdx, :], CurrentLabels,
						                               Mode='Random')

				# print('Not enought samples in:' + repr(i))

				# center loss
				if UseLoss['Center']:
					CenterLoss += self.CenterLoss(input['CascadeEmbed'][i][CurrentsIdx, :],
					                              EmbeddingCenters[i](CurrentLabels))

			# losses = GreedyLoss + ClassLosses  + ProbLoss + CascadeEmbeddLoss
			Loss = CascadedClassLoss + CenterLoss + MVloss + OrdinalLoss + MlLoss + SphereLoss + RegressionLoss

			Loss /= len(input['CascadeClassScores'])

			AllLosses = {}
			AllLosses['CascadeClass'] = CascadedClassLoss / len(input['CascadeClassScores'])
			AllLosses['Center'] = CenterLoss / len(input['CascadeClassScores'])
			AllLosses['MVloss'] = MVloss / len(input['CascadeClassScores'])
			AllLosses['Ordinal'] = OrdinalLoss / len(input['CascadeClassScores'])
			AllLosses['MlLoss'] = MlLoss / len(input['CascadeClassScores'])
			AllLosses['Sphere'] = SphereLoss / len(input['CascadeClassScores'])
			AllLosses['Regress'] = RegressionLoss / len(input['CascadeClassScores'])

			return Loss, AllLosses