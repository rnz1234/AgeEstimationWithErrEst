import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, MaxPool2d, BatchNorm2d, Dropout, Conv1d, MaxPool1d, Sigmoid, LSTM, \
	AvgPool1d  # ,TransformerEncoder, TransformerEncoderLayer

import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
import cv2
import albumentations as A

from Models.UnifiedClassificaionAndRegressionAgeModel import FeatureExtractionVgg16
from Models.inception_resnet_v1 import InceptionResnetV1
import PIL
from torchvision.transforms import transforms
from Losses.AngleLinear import AngleLinear
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from PIL import Image
# from utils.augmentations import Lighting, RandAugment
import random
from Models.keller_transformer import Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, \
	TransformerEncoderLayer


def sigmoid(X):
	return 1 / (1 + np.exp(-X))


def ComputeError(Error, AgeIntareval=1):
	Result = {}

	Result['Errors'] = 100 * (Error > 0).sum() / float(Error.shape[0])
	Result['Error1'] = 100 * (Error == 1).sum() / float(Error.shape[0])
	Result['Error2'] = 100 * (Error == 2).sum() / float(Error.shape[0])
	Result['Error3'] = 100 * (Error == 3).sum() / float(Error.shape[0])
	Result['Error4'] = 100 * (Error == 4).sum() / float(Error.shape[0])
	Result['Error5'] = 100 * (Error == 5).sum() / float(Error.shape[0])

	Result['AverageLoss'] = Error.mean()
	Result['AverageLoss'] *= AgeIntareval

	return Result


def NormalizeImages(x):
	# x = (x-127)/255

	Mean = [0.485, 0.456, 0.406]
	Std = [0.229, 0.224, 0.225]

	x /= 255
	for i in range(3):
		x[:, i, :, :] = (x[:, i, :, :] - Mean[i]) / Std[i]
	# Result = x / (255.0/2)

	# ransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	return x


def EvaluateNet(net, DataLoader, device, Mode):
	TestLabels = None

	DataSize = len(DataLoader.dataset)
	batch_size = DataLoader.batch_size

	NoElements = 0
	with torch.no_grad():
		# for k in range(0, data.shape[0], StepSize):
		for k, Data in enumerate(DataLoader):

			Input = Data['Images'].squeeze()
			#
			# plt.imshow(Input[0, 0,].transpose(0, 2));plt.show()
			# (Input[1, 0,] - Input[0, 2,]).abs().sum()
			# Input.shape

			if Input.ndim == 5:
				if Input.shape[0] % 6 > 0:
					Input = Input[0:int(Input.shape[0] / 6) * 6, ]

				NoElements += Input.shape[0]
				# print(repr(NoElements))

				Input = np.reshape(Input.transpose(0, 1),
				                   (Input.shape[0] * Input.shape[1],
				                    Input.shape[2], Input.shape[3], Input.shape[4]),
				                   order='F')
			# (Input[99,] - Input[100,]).abs().sum()
			# (Input[23*3+0, ] -  Data['Images'][23, 1,]).abs().sum()

			# ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])

			# Input = Data['Images'][:, 0, ]
			# net.AugmentNo = 1

			Input = Input.to(device)

			a = net(Input, Mode)

			# (a['OrdinalClass'] > 0).sum(1)
			# ((a['OrdinalClass'] > 0).sum(1).cpu() - np.round(Data['Labels'].squeeze()))

			del Data['Images']
			a = {**a, **Data}

			if k == 0:
				keys = list(a.keys())
				EmbA = dict()
				for key in keys:
					if type(a[key]) == list:
						EmbA[key] = []
						for x in a[key]:
							EmbA[key].append(x)
						continue

					if type(a[key]) == torch.Tensor:

						if a[key].ndim == 1:
							EmbA[key] = np.zeros((DataSize), dtype=np.float32)
						if a[key].ndim == 2:
							EmbA[key] = np.zeros((DataSize, a[key].shape[1]), dtype=np.float32)
						if a[key].ndim == 3:
							EmbA[key] = np.zeros((DataSize, a[key].shape[1], a[key].shape[2]), dtype=np.float32)

						continue

			for key in keys:
				if type(a[key]) == list:

					try:
						if k > 0:
							for i in range(len(a[key])):
								EmbA[key][i] = torch.cat((EmbA[key][i], a[key][i]), 0)
						continue
					except:
						ss = 5

				if type(a[key]) == torch.Tensor:
					try:
						EmbA[key][k * batch_size:(k * batch_size + a[key].shape[0])] = a[key].cpu()
						continue
					except:
						ss = 5
						print('Error in EvaluateNet')

	if Input.ndim == 5:
		for key in keys:
			if type(a[key]) == torch.Tensor:
				EmbA[key] = EmbA[key][0:NoElements, ]

	return EmbA


def ShowRowImages(image_data):
	fig = plt.figure(figsize=(1, image_data.shape[0]))
	grid = ImageGrid(fig, 111,  # similar to subplot(111)
	                 nrows_ncols=(1, image_data.shape[0]),  # creates 2x2 grid of axes
	                 axes_pad=0.1,  # pad between axes in inch.
	                 )
	# for ax, im in zip(grid, image_data):
	for ax, im in zip(grid, image_data):
		if im.shape[2] == 1:
			ax.imshow(im, cmap='gray')
		if im.shape[2] == 3:
			ax.imshow(im)
	plt.show()


def ShowTwoRowImages(image_data1, image_data2):
	fig = plt.figure(figsize=(2, image_data1.shape[0]))
	grid = ImageGrid(fig, 111,  # similar to subplot(111)
	                 nrows_ncols=(2, image_data1.shape[0]),  # creates 2x2 grid of axes
	                 axes_pad=0.1,  # pad between axes in inch.
	                 )
	# for ax, im in zip(grid, image_data):
	for ax, im in zip(grid, image_data1):
		# Iterating over the grid returns the Axes.
		if im.shape[2] == 1:
			ax.imshow(im, cmap='gray')
		if im.shape[2] == 3:
			ax.imshow(im)

	for i in range(image_data2.shape[0]):
		# Iterating over the grid returns the Axes.
		if im.shape[2] == 1:
			grid[i + image_data1.shape[0]].imshow(image_data2[i], cmap='gray')
		if im.shape[2] == 3:
			grid[i + image_data1.shape[0]].imshow(image_data2[i])
	plt.show()


def imshow(img):
	img = img / 2 + 0.5  # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


class ClassificationCnn(nn.Module):

	def __init__(self, CascadeSupport, K):
		super(ClassificationCnn, self).__init__()

		self.CascadeSupport = CascadeSupport

		self.BaseFC = Linear(512, K)

		self.fc_class1 = Linear(K, CascadeSupport)

		self.ordinal_fc1 = Linear(K, CascadeSupport)
		self.ordinal_fc1_bias = nn.Parameter(torch.zeros(CascadeSupport).float())

		self.regressionl_fc1 = Linear(K, 1)
		self.regression_activ = nn.Hardtanh()

		self.AngleLinear = AngleLinear(K, CascadeSupport)

	def FreezeBaseFC(self, OnOff):
		for param in self.BaseFC.parameters():
			param.requires_grad = not OnOff

	def forward(self, BaseEmbedding, DropoutP=0):
		# Base = self.BaseFC(BaseEmbedding)
		# Base = F.leaky_relu(Base)
		# Base  = self.BaseBN(Base)
		# Base = F.normalize(Base, dim=1, p=2)
		# BaseEmbedding = nn.Dropout(DropoutP)(Base)

		# classifcation
		Classification = self.fc_class1(BaseEmbedding)

		# ordinal
		# Ordinal  = self.ordinal_fc0(BaseD)
		# Ordinal  = F.leaky_relu(Ordinal)
		# Ordinal  = self.ordinal_fc1(Ordinal)
		Ordinal = self.ordinal_fc1(BaseEmbedding)
		Ordinal += self.ordinal_fc1_bias

		# extended ordinal

		# regression
		# Regression = self.regressionl_fc1(BaseD)
		# Regression = self.regressionl_fc0(BaseD)
		# Regression = F.leaky_relu(Regression)
		# Regression = nn.Dropout(DropoutP)(Regression)
		Regression = self.regressionl_fc1(BaseEmbedding)
		# Regression = self.regression_activ(Regression)

		# sphereloss
		# Sphere = self.AngleLinear(BaseD)

		Result = {}
		Result['Class'] = Classification
		# Result['Embed'] = Base
		Result['OrdinalClass'] = Ordinal
		# Result['Sphere'] = Sphere
		Result['Regress'] = Regression

		return Result


class FaceClassification(nn.Module):

	def __init__(self, NumClasses, CascadeSupport, CascadeSkip, DropoutP=0.5, K=256, AugmentNo=1):
		super(FaceClassification, self).__init__()

		K = int(K)
		self.NumClasses = int(NumClasses)
		self.CascadeSupport = int(CascadeSupport)
		self.Labels = range(NumClasses)
		self.CascadeSkip = CascadeSkip
		self.CascadeSize = round(self.NumClasses / CascadeSkip)
		self.AugmentNo = AugmentNo

		# ----    classifcation layers

		# basenet
		FcaeNetDropout = 0
		self.num_features = 512
		self.base_net = InceptionResnetV1(pretrained='vggface2', dropout_prob=FcaeNetDropout)

		# self.num_features = 4096
		# self.base_net = FeatureExtractionVgg16()

		# age classifcation
		self.fc_class = Linear(self.num_features, K)
		self.fc_class1 = Linear(K, NumClasses)
		self.class_head = Linear(K, NumClasses)

		# ordinal
		self.ordinal_fc = Linear(self.num_features, K)
		self.ordinal_fc1 = Linear(K, NumClasses, bias=False)
		self.ordinal_fc1_bias = nn.Parameter(torch.zeros(NumClasses).float())
		self.ordinal_fc2 = Linear(K, 1, bias=False)

		# regression
		self.fc_regress = Linear(self.num_features, K)

		self.regression_fc = Linear(self.num_features, K)
		self.regressionl_fc1 = Linear(K, 1)

		self.regression_heads = []
		self.centers = []

		for i in range(self.NumClasses):
			self.regression_heads.append(Linear(K, 1))
			self.centers.append(i)

		self.regression_heads = nn.ModuleList(self.regression_heads)

		self.Embedding = nn.Embedding(NumClasses, K)

		# --------------------------   DETR  ----------------------------------
		TransformerLayersNo = 2
		TransformerHeadsNo = 2
		self.DetrTransformer = Transformer(
			d_model=K, dropout=0.1, nhead=TransformerHeadsNo,
			dim_feedforward=K,
			num_encoder_layers=TransformerLayersNo,
			num_decoder_layers=TransformerLayersNo,
			normalize_before=False, return_intermediate_dec=False)

		# cascade layers
		self.cascade_cnn = nn.ModuleList()
		# self.CascadeAngularLoss = nn.ModuleList()
		self.CascadeEmbedding = nn.ModuleList()
		for i in range(int(np.ceil(CascadeSupport))):
			self.cascade_cnn.append(ClassificationCnn(self.CascadeSupport, int(K)))
			self.CascadeEmbedding.append(nn.Embedding(self.CascadeSupport, int(K)))

	# self.HalfCascade = int(self.CascadeSupport / 2)
	# self.MinimalActivation = -10

	def FreezeBaseCnn(self, OnOff):
		for param in self.base_net.parameters():
			param.requires_grad = not OnOff

	def FreezeFaceNetFC(self, OnOff):
		for param in self.base_net.last_linear.parameters():
			param.requires_grad = not OnOff

		for param in self.base_net.last_bn.parameters():
			param.requires_grad = not OnOff

	def FreezeClassEmbeddFC(self, OnOff):
		for param in self.fc_class.parameters():
			param.requires_grad = not OnOff

	def FreezeAgeClassificationFC(self, OnOff):
		for param in self.fc_class1.parameters():
			param.requires_grad = not OnOff

	def ApplyAngularLoss(self, x, labels):
		return self.AngularLoss.forward(x, labels)

	def FreezeEmbedding(self, OnOff):
		for param in self.Embedding.parameters():
			param.requires_grad = not OnOff

	def FreezeOrdinalFC(self, OnOff):
		for param in self.ordinal_fc.parameters():
			param.requires_grad = not OnOff

	def FreezeOrdinalLayers(self, OnOff):
		for param in self.ordinal_fc.parameters():
			param.requires_grad = not OnOff

		for param in self.ordinal_fc1.parameters():
			param.requires_grad = not OnOff

		self.ordinal_fc1_bias.requires_grad = not OnOff

		for param in self.ordinal_bn.parameters():
			param.requires_grad = not OnOff

	def FreezeHeatmapLayers(self, OnOff):
		for param in self.HeatmapFC.parameters():
			param.requires_grad = not OnOff

		for param in self.HeatmapFC1.parameters():
			param.requires_grad = not OnOff

	def FreezeEthnicityLayers(self, OnOff):
		for param in self.fc_ethnicity.parameters():
			param.requires_grad = not OnOff

		for param in self.fc_ethnicity1.parameters():
			param.requires_grad = not OnOff

	def FreezeGenderLayers(self, OnOff):
		for param in self.fc_gender.parameters():
			param.requires_grad = not OnOff

		for param in self.fc_gender1.parameters():
			param.requires_grad = not OnOff

	def FreezeOrdinalEncoder(self, OnOff):
		for param in self.OrdinalEncoder.parameters():
			param.requires_grad = not OnOff

	def forward(self, Images, Input=None, Labels=None, DropoutP=0.5):

		unpacked_input = Images.view(Images.shape[0] * Images.shape[1], Images.shape[2], Images.shape[3],
		                             Images.shape[4])

		BaseEmbedding = self.base_net(unpacked_input)
		BaseEmbedding = F.leaky_relu(BaseEmbedding)

		# age classification
		ClassEmbed = self.fc_class(BaseEmbedding)
		ClassEmbed = F.normalize(ClassEmbed, dim=1, p=2)  # L2 normalization

		EmbeddingRep = self.Embedding.weight.data.unsqueeze(1)
		EmbeddingRep = EmbeddingRep.repeat(1, BaseEmbedding.shape[0], 1)

		ClassEmbed = ClassEmbed.t().contiguous().t()
		ClassEmbed = ClassEmbed.reshape(
			(int(ClassEmbed.shape[0] / self.AugmentNo), int(self.AugmentNo), ClassEmbed.shape[1]))

		AgeClass = ClassEmbed[:, 0, ]

		# ordinal binary
		OrdinalClass = self.ordinal_fc(BaseEmbedding)
		OrdinalClass = F.leaky_relu(OrdinalClass)
		OrdinalClass = F.normalize(OrdinalClass, dim=1, p=2)

		OrdinalEmbed = OrdinalClass

		EmbeddingRep = self.Embedding.weight.data.unsqueeze(1).repeat(1, int(
			BaseEmbedding.shape[0] / self.AugmentNo), 1)
		EmbeddingRep = F.normalize(EmbeddingRep, dim=2, p=2)  # L2 normalization

		OrdinalClass = OrdinalClass.t().contiguous().t()
		OrdinalClass = OrdinalClass.reshape(
			(int(OrdinalClass.shape[0] / self.AugmentNo), int(self.AugmentNo), OrdinalClass.shape[1]))
		OrdinalClass = torch.transpose(OrdinalClass, 0, 1)
		OrdinalClass = F.normalize(OrdinalClass, dim=2, p=2)  # L2 normalization

		h, memory = self.DetrTransformer(src=OrdinalClass, mask=None,
		                                 query_embed=self.Embedding.weight.data, pos_embed=None)
		# OrdinalEmbed = memory.mean(0)
		# OrdinalEmbed = F.normalize(OrdinalEmbed, dim=1, p=2)  # L2 normalization

		OrdinalClass = self.ordinal_fc2(h).squeeze().transpose(1, 0)
		OrdinalClass += self.ordinal_fc1_bias

		# center loss normalization
		self.Embedding.weight.data = F.normalize(self.Embedding.weight.data, p=2, dim=1)

		# extended ordinal
		# OrdinalProbs = nn.Sigmoid()(OrdinalClass)
		# OrdinalClassificationProbs = -(OrdinalProbs[:, 1:] - OrdinalProbs[:, 0:-1])
		# OrdinalClassificationProbs = F.softmax(OrdinalClassificationProbs, 1)

		# output
		Result = dict()
		Result['Class'] = AgeClass
		Result['ClassEmbed'] = ClassEmbed

		Result['OrdinalClass'] = OrdinalClass
		# Result['OrdinalEmbed'] = OrdinalEmbed
		# Result['ExtendedOrdinalClass'] = OrdinalClassificationProbs

		return Result


def Create_Training_Test_Sets(Ids, TrainingRatio):
	# compute Unique Ids and images per ID
	UniqueIds, LinearIds = np.unique(Ids, return_inverse=True)
	ImagesPerID = []
	for i in range(UniqueIds.shape[0]):
		idx = np.where(Ids == UniqueIds[i])[0];
		ImagesPerID.append(idx)

	# find training set
	ShuffledUniqueIds = np.arange(UniqueIds.shape[0])
	np.random.shuffle(ShuffledUniqueIds)

	TrainIdSet = []
	TrainSamples = []
	NumTrainSamples = int(TrainingRatio * Ids.shape[0])
	CurrNumTrainSamples = 0
	for i in ShuffledUniqueIds:

		# add set
		TrainIdSet.append(i)

		# add samples in set
		TrainSamples.append(ImagesPerID[i])

		CurrNumTrainSamples += ImagesPerID[i].shape[0]
		if CurrNumTrainSamples >= NumTrainSamples:
			break

	TrainIdSet = np.asarray(TrainIdSet)
	TrainSamples = np.concatenate(TrainSamples)

	# find test set
	TestIdSet = np.setdiff1d(np.arange(UniqueIds.shape[0]), TrainIdSet)

	TestSamples = np.setdiff1d(np.arange(Ids.shape[0]), TrainSamples)

	Result = {}
	Result['ImagesPerID'] = ImagesPerID
	Result['UniqueIds'] = UniqueIds

	Result['TrainIdSet'] = TrainIdSet
	Result['TestIdSet'] = TestIdSet

	Result['LinearIds'] = LinearIds

	Result['TrainSamples'] = TrainSamples
	Result['TestSamples'] = TestSamples

	return Result


def HardMining(net, criterion, Images, Labels, HardRatio, device, MaxErrorImagesNo, StepSize, CnnMode):
	with torch.no_grad():

		if CnnMode == 'Classify':
			Embed = EvaluateNet(net, Images, device, StepSize, Mode=CnnMode)
			loss = criterion(torch.from_numpy(Embed['Class']), Labels.round().long())
			idx = torch.sort(loss, descending=True)[1]
			idx = idx[0:int(idx.shape[0] * HardRatio)]
			idx = np.random.permutation(idx)

			Result = {}
			Result['HardIdx'] = idx

		if CnnMode == 'Regress':
			Embed = EvaluateNet(net, Images, device, StepSize, Mode=CnnMode)
			loss = criterion(torch.from_numpy(Embed), Labels)

		if CnnMode == 'Cascade':
			Embed = EvaluateNet(net, Images, device, StepSize, Mode='Cascade_test')
			ErrorIdx = np.where((Embed['CascadeClass'] - Labels.numpy().round()) > 0)[0]
			NonErrorIdx = np.where((Embed['CascadeClass'] - Labels.numpy().round()) == 0)[0]

			if ErrorIdx.shape[0] > MaxErrorImagesNo:
				ErrorIdx = ErrorIdx[0:MaxErrorImagesNo]

			NonErrorIdx = NonErrorIdx[np.random.randint(NonErrorIdx.shape[0],
			                                            size=min(int((1.0 / HardRatio) * ErrorIdx.shape[0]),
			                                                     NonErrorIdx.shape[0]))]

			idx = np.concatenate((ErrorIdx, NonErrorIdx))
			idx = np.random.permutation(idx)

			Result = {}
			Result['HardIdx'] = idx
	# Result['HardIdx'] = range(Images.shape[0])

	return Result


# params = add_weight_decay(net, 2e-5)
# sgd = torch.optim.SGD(params, lr=0.05)
def add_weight_decay(net, l2_value, skip_list=()):
	decay, no_decay = [], []

	for name, param in net.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights

		if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
			no_decay.append(param)
		else:
			decay.append(param)
	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def ComputeErrorHistogram(Labels, Errors):
	if type(Labels) == torch.Tensor:
		SortedLabels = torch.unique(Labels)

	if type(Labels) == np.ndarray:
		SortedLabels = np.unique(Labels)

	ErrorHist = np.zeros((SortedLabels.shape), dtype=np.float)
	for i in range(SortedLabels.shape[0]):

		# get all of the results of a particular Label
		idx = np.where(Labels == SortedLabels[i])[0]

		if idx.size == 0:
			continue

		ErrorHist[i] = Errors[idx].mean()

	Result = {}

	Result['ErrorHist'] = ErrorHist
	Result['SortedLables'] = SortedLabels

	return Result

# def AnalyzeCascadeErrors(Response,Labesl)
