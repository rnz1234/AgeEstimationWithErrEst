import math
import os
import random

import GPUtil
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Datasets.Morph2.DataParser import DataParser
from Datasets.Morph2.Morph2ClassifierDataset import Morph2ClassifierDataset
from Losses import WeightedBinaryCrossEntropyLoss
from Optimizers.RangerLars import RangerLars
from Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Training.train_keller_ordinal import train_keller_ordinal
from face_classes import FaceClassification

torch.backends.cudnn.deterministic = True  # needed
torch.backends.cudnn.benchmark = True
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
GPUtil.showUtilization()

min_age = 15
max_age = 80
age_interval = 1
batch_size = 18
num_epochs = 30
random_split = False
num_copies = 6
mid_feature_size = 512

num_classes = int((max_age - min_age) / age_interval + 1)

# Load data
data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
data_parser.initialize_data()

x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test,
if random_split:
	all_images = np.concatenate((x_train, x_test), axis=0)
	all_labels = np.concatenate((y_train, y_test), axis=0)

	x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.20, random_state=42)

train_ds = Morph2ClassifierDataset(
	x_train,
	y_train,
	min_age,
	age_interval,
	transforms.Compose([
		transforms.RandomResizedCrop(160, (0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomApply([transforms.ColorJitter(
			brightness=0.1,
			contrast=0.1,
			saturation=0.1,
			hue=0.1
		)], p=0.5),
		transforms.RandomApply([transforms.RandomAffine(
			degrees=10,
			translate=(0.1, 0.1),
			scale=(0.9, 1.1),
			shear=5,
			resample=Image.BICUBIC
		)], p=0.5),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		transforms.RandomErasing(p=0.5)
	]),
	copies=num_copies
)

test_ds = Morph2ClassifierDataset(
	x_test,
	y_test,
	min_age,
	age_interval,
	transform=transforms.Compose([
		# transforms.Resize(224),
		# transforms.ToTensor(),
		# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		transforms.RandomResizedCrop(160, (0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomApply([transforms.ColorJitter(
			brightness=0.1,
			contrast=0.1,
			saturation=0.1,
			hue=0.1
		)], p=0.5),
		transforms.RandomApply([transforms.RandomAffine(
			degrees=10,
			translate=(0.1, 0.1),
			scale=(0.9, 1.1),
			shear=5,
			resample=Image.BICUBIC
		)], p=0.5),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		transforms.RandomErasing(p=0.5)
	]),
	copies=num_copies
)

image_datasets = {
	'train': train_ds,
	'val': test_ds
}

data_loaders = {
	'train': DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True),
	'val': DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# MseLoss = nn.MSELoss().to(device)
# CentLoss = CenterLoss(reduction='mean').to(device)
# mean_var_criterion = MeanVarianceLoss(0, num_classes, device, lambda_mean=0.2, lambda_variance=0.05).to(device)
# MetricLearnLoss = MetricLearningLoss(LabelsDistT=3, reduction='mean').to(device)
# CeLoss = nn.CrossEntropyLoss().to(device)
# AngularleLoss = AngleLoss().to(device)
# BCEloss = nn.BCEWithLogitsLoss().to(device)
# CELogProb = nn.NLLLoss().to(device)
# KLDiv = nn.KLDivLoss(reduction='batchmean').to(device)
# Heatmap10 = Heatmap1Dloss(device, NumLabes=num_classes, sigma=10.0)
WeightedBceLoss = WeightedBinaryCrossEntropyLoss(num_classes, device, reduction='mean', binary_loss_type='BCE')

net = FaceClassification(
	NumClasses=num_classes, CascadeSupport=15, CascadeSkip=5,
	K=mid_feature_size, DropoutP=0.5, AugmentNo=num_copies)
net.to(device)

net.FreezeBaseCnn(True)
#
# optimizer = RangerLars(net.parameters(), lr=1e-4)
# cosine_scheduler = CosineAnnealingLR(
# 	optimizer,
# 	T_max=num_epochs
# )
# scheduler = GradualWarmupScheduler(
# 	optimizer,
# 	multiplier=1,
# 	# total_epoch=num_epochs // 10,
# 	total_epoch=5,
# 	after_scheduler=cosine_scheduler
# )

optimizer = torch.optim.Adam(net.parameters(),  lr=5e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
# scheduler = StepLR(optimizer, step_size=5, gamma=math.sqrt(0.1))

### Train ###
writer = SummaryWriter('logs/Morph2/transformer/keller_layers_2_heads_2_adam_5e4_decay_1e5_ReduceLROnPlateau_batch_18_copies_6_mid_feature_size_512_0rdinal_imsize_160_facenet_unfreeze_5')
# writer = None

best_model = train_keller_ordinal(
	net,
	WeightedBceLoss,
	optimizer,
	scheduler,
	data_loaders,
	dataset_sizes,
	device,
	writer,
	num_classes,
	min_age,
	num_epochs=num_epochs
)

print('saving best model')

model_path = 'weights/Morph2/keller_transformer/keller_layers_2_heads_2_adam_5e4_decay_1e5_ReduceLROnPlateau_batch_18_copies_6_mid_feature_size_512_0rdinal_imsize_160_facenet_unfreeze_5'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
torch.save(best_model.state_dict(), FINAL_MODEL_FILE)

print('fun fun in the sun, the training is done :)')
