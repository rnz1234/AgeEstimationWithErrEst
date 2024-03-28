# adapted from https://github.com/foamliu/InsightFace-v2/blob/e07b738adecb69b81ac9b8750db964cee673e175/models.py#L327
import math

import torch
from torch import nn as nn
from torch.nn import Linear, functional as F
from torchvision.models import resnet34, vgg16


class BaseRecogClassifier(nn.Module):
	def __init__(self, num_classes):
		super(BaseRecogClassifier, self).__init__()


		# self.base_net = InceptionResnetV1(pretrained='vggface2')
		# self.fc_class = Linear(512, k)

		# self.base_net = resnet34(pretrained=True)
		# self.base_net.fc = Linear(self.base_net.fc.in_features, k)

		self.base_net = vgg16(pretrained=True)
		#self.base_net = vgg16(weights='DEFAULT')


		# self.FaceNet.classifier[6] = nn.Linear(4096, 512)

		num_features = 84*7*7 #self.base_net.classifier[6].in_features
		# Add a new fully connected layer with the desired number of output units
		#del self.base_net.classifier[2:]

		self.AdaptiveAvgPoolLayer = nn.AdaptiveAvgPool2d(output_size=(7, 7))
		self.ReducLayer = nn.Conv2d(in_channels=512, out_channels=84, kernel_size=1)
		self.FlatLayer = nn.Flatten()
		
		# face classification
		# self.FaceClassFC = #AngleLinear(512, NumFaceClasses)
		self.FaceClassFC = nn.Linear(num_features, num_classes)
		self.BaseEmbAct = nn.LeakyReLU()
		self.DropoutLayer = nn.Dropout(p=0.5)
		self.FaceBatchNorm1d = nn.BatchNorm1d(num_features, affine=False)

	
		# self.base_net.classifier = nn.Sequential(
		# 	nn.Dropout(p=0.5),
		# 	nn.Linear(in_features=25088, out_features=4096, bias=True),
		# 	nn.ReLU(inplace=True),
		# 	nn.BatchNorm1d(4096),
		# )

		# self.fc_head = nn.Sequential(
		# 	nn.Dropout(p=0.5),
		# 	nn.Linear(in_features=4096, out_features=2048),
		# 	nn.ReLU(inplace=True),
		# 	#nn.Dropout(p=0.5),
		# 	nn.Linear(in_features=2048, out_features=1024),
		# 	nn.ReLU(inplace=True),
		# 	#nn.Dropout(p=0.5),
		# 	nn.Linear(in_features=1024, out_features=num_classes)
		# )
		
		

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze

	def forward(self, input_images, device=torch.device("cuda:0")):
		#x = self.base_net(input_images)
		#output = self.fc_head(x)

		PreBaseEmbedding = self.base_net.features(input_images)

		PreBaseEmbedding1 = self.AdaptiveAvgPoolLayer(PreBaseEmbedding)
		PreBaseEmbedding2 = self.ReducLayer(PreBaseEmbedding1)
		BaseEmbedding = self.FlatLayer(PreBaseEmbedding2)

		IdEmbed = self.BaseEmbAct(BaseEmbedding)
		IdEmbed = self.FaceBatchNorm1d(IdEmbed)
		IdEmbed = self.DropoutLayer(IdEmbed)
		output = self.FaceClassFC(IdEmbed)

		return output