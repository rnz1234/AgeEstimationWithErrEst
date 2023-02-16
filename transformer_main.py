import os
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Datasets.AFAD.AFADClassifierDataset import AFADClassifierDataset
from Datasets.Morph2.DataParser import DataParser, DataParserWithLimit
from Datasets.Morph2.Morph2ClassifierDataset import Morph2ClassifierDataset
from Losses.MeanVarianceLoss import MeanVarianceLoss
from Models.JoinedTransformerModel import JoinedTransformerModel
from Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel
from Models.transformer import *
from Models.unified_transformer_model import AgeTransformer
from Optimizers.RangerLars import RangerLars
from Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Training.train_unified_model_iter import train_unified_model_iter, train_unified_model_with_err_est_iter


DEBUG = True #True
MULTI_GPU = True
SMALL_DATA = True

def get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size):
	#pretrained_model = UnifiedClassificaionAndRegressionAgeModel(7, 10, 15, 80)
	
	pretrained_model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age)
	#pretrained_model_path = 'weights/Morph2/unified/RangerLars_lr_5e4_4096_epochs_60_batch_32_mean_var_vgg16_pretrained_recognition_bin_10_more_augs_RandomApply_warmup_cosine_recreate'
	
	# # Use this for starting from backbone weights trained for e2e task with transformer:
	# pretrained_model_path = 'weights/Morph2/transformer/encoder/time_10_01_2022_09_08_38bin_1_layers_4_heads_4_lr0.0003_batch_8_copies_10_mid_feature_size_1024_augs_at_val_context4'
	
	# Use this for starting from backbone weights trained for e2e task with no transformer:
	pretrained_model_path = 'weights/Morph2/unified/iter/RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_1_2'

	
	# pretrained_model = UnifiedClassificaionAndRegressionAgeModel(4, 8, 12, 43)
	# pretrained_model_path = 'weights/AFAD/unified/iter/RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_8_stronger_augs_2'

	pretrained_model_file = os.path.join(pretrained_model_path, "weights.pt")
	pretrained_model.load_state_dict(torch.load(pretrained_model_file), strict=False)

	num_features = pretrained_model.num_features
	backbone = pretrained_model.base_net
	backbone.train()
	backbone.to(device)

	# backbone = InceptionResnetV1(pretrained='vggface2')
	# num_features = 512

	transformer = TransformerModelWithErrEstimator( #TransformerModel(
		age_interval, min_age, max_age,
		mid_feature_size, mid_feature_size,
		num_outputs=num_classes,
		n_heads=4, n_encoders=4, dropout=0.3,
		mode='context').to(device)
	age_transformer = AgeTransformer(backbone, transformer, num_features, mid_feature_size)

	using_multi_gpu = False
	if MULTI_GPU:
		if torch.cuda.device_count() > 1:
			print("Using multiple GPUs (" + str(torch.cuda.device_count()) + ")")
			age_transformer = torch.nn.DataParallel(age_transformer)
			using_multi_gpu = True
	age_transformer.to(device)

	return age_transformer, using_multi_gpu


def get_joined_model(device, num_classes, age_interval, min_age, max_age, mid_feature_size):
	model = JoinedTransformerModel(num_classes, age_interval, min_age, max_age, device, mid_feature_size)
	model.to(device)
	model.train()

	return model


if __name__ == "__main__":
	seed = 42
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	torch.cuda.empty_cache()

	torch.backends.cudnn.benchmark = True

	min_age = 15 #Morph
	max_age = 80 #Morph
	age_interval = 1 #Morph
	# min_age = 12 #AFAD
	# max_age = 43 #AFAD
	# age_interval = 8  # AFAD
	batch_size = 16 #8
	# num_epochs = 60
	num_iters = int(1.8e5) #int(1.5e5)
	random_split = False #True
	num_copies = 10
	mid_feature_size = 1024

	########################################
	## Err Est
	err_est_bin_range_lo = 4
	########################################

	num_classes = int((max_age - min_age) / age_interval + 1)

	# Load data
	#data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
	data_parser = DataParserWithLimit('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5', SMALL_DATA)
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
			transforms.RandomResizedCrop(224, (0.9, 1.0)),
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
			transforms.RandomResizedCrop(224, (0.9, 1.0)),
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

	# train_ds = AFADClassifierDataset(
	# './Datasets/AFAD/aligned_data/afad_train.h5',
	# min_age=min_age,
	# max_age=max_age,
	# age_interval=age_interval,
	# 	transform=transforms.Compose([
	# 		transforms.RandomResizedCrop(224, (0.9, 1.0)),
	# 		transforms.RandomHorizontalFlip(),
	# 		transforms.RandomApply([transforms.ColorJitter(
	# 			brightness=0.2,
	# 			contrast=0.2,
	# 			saturation=0.2,
	# 			hue=0.2
	# 		)], p=0.5),
	# 		transforms.RandomApply([transforms.RandomAffine(
	# 			degrees=20,
	# 			translate=(0.2, 0.2),
	# 			scale=(0.8, 1.2),
	# 			shear=10,
	# 			resample=Image.BICUBIC
	# 		)], p=0.5),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	# 		transforms.RandomErasing(p=0.5)
	# 	]),
	# 	copies=num_copies
	# )
	#
	# test_ds = AFADClassifierDataset(
	# 	'./Datasets/AFAD/aligned_data/afad_test.h5',
	# 	min_age=min_age,
	# 	max_age=max_age,
	# 	age_interval=age_interval,
	# 	transform=transforms.Compose([
	# 		transforms.RandomResizedCrop(224, (0.9, 1.0)),
	# 		transforms.RandomHorizontalFlip(),
	# 		transforms.RandomApply([transforms.ColorJitter(
	# 			brightness=0.2,
	# 			contrast=0.2,
	# 			saturation=0.2,
	# 			hue=0.2
	# 		)], p=0.5),
	# 		transforms.RandomApply([transforms.RandomAffine(
	# 			degrees=20,
	# 			translate=(0.2, 0.2),
	# 			scale=(0.8, 1.2),
	# 			shear=10,
	# 			resample=Image.BICUBIC
	# 		)], p=0.5),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	# 		transforms.RandomErasing(p=0.5)
	# 	]),
	# 	copies=num_copies
	# )

	image_datasets = {
		'train': train_ds,
		'val': test_ds
	}

	if DEBUG:
		num_workers = 0
	else:
		num_workers = 4

	data_loaders = {
		'train': DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True, drop_last=True),
		'val': DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False, drop_last=True)
	}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	# create model and parameters
	model, using_multi_gpu = get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size)
	# model = get_joined_model(device, num_classes, age_interval, min_age, max_age, mid_feature_size)
	

	model.module.FreezeBaseCnn(True)

	criterion_reg = nn.MSELoss().to(device)
	criterion_cls = torch.nn.CrossEntropyLoss().to(device)
	mean_var_criterion = MeanVarianceLoss(0, num_classes, device, lambda_mean=0.2, lambda_variance=0.05).to(device)

	lr = 0.00028 #3e-4
	optimizer = RangerLars(model.parameters(), lr=lr)#lr=0.0003) #1e-3)

	num_epochs = int(num_iters / len(data_loaders['train'])) + 1

	cosine_scheduler = CosineAnnealingLR(
		optimizer,
		T_max=num_iters
	)
	scheduler = GradualWarmupScheduler(
		optimizer,
		multiplier=1,
		total_epoch=10000,
		# total_epoch=5,
		after_scheduler=cosine_scheduler
	)

	###################################################
	## Err Est 

	criterion_err_est = nn.BCEWithLogitsLoss().to(device)

	# optimizer_err_est = RangerLars(model.parameters(), lr=lr)#lr=0.0003) #1e-3)

	# cosine_scheduler_err_est = CosineAnnealingLR(
	# 	optimizer_err_est,
	# 	T_max=num_iters
	# )
	# scheduler_err_est = GradualWarmupScheduler(
	# 	optimizer_err_est,
	# 	multiplier=1,
	# 	total_epoch=10000,
	# 	# total_epoch=5,
	# 	after_scheduler=cosine_scheduler_err_est
	# )
	###################################################

	from datetime import datetime
	cur_time = datetime.now()
	cur_time_str = cur_time.strftime("time_%d_%m_%Y_%H_%M_%S")

	### Train ###
	# import pdb
	# pdb.set_trace()
	writer = SummaryWriter('logs/Morph2/transformer/encoder/' + cur_time_str + 'bin_1_layers_4_heads_4_lr' + str(lr) + '_batch_' + str(batch_size) + '_iters_' + str(num_iters) + '_copies_10_mid_feature_size_1024_augs_at_val_context4') #_context_true_iter_warmup_10000_amp_batchnorm_after_encoder_iter_15e5_no_encoder_only_mean') #_RS')
	# writer = None

	model_path = 'weights/Morph2/transformer/encoder/' + cur_time_str + 'bin_1_layers_4_heads_4_lr' + str(lr) + '_batch_' + str(batch_size) + '_iters_' + str(num_iters) + '_copies_10_mid_feature_size_1024_augs_at_val_context4' #_context_true_iter_warmup_10000_amp_batchnorm_after_encoder_iter_15e5_no_encoder_only_mean_RS'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	# model_path = None

	best_model = train_unified_model_with_err_est_iter( #train_unified_model_iter(
		model,
		criterion_reg,
		criterion_cls,
		mean_var_criterion,
		optimizer,
		scheduler,
		criterion_err_est,
		# optimizer_err_est,
		# scheduler_err_est,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		model_path,
		num_classes,
		num_epochs=num_epochs,
		validate_at_k=1000,
		err_est_bin_range_lo=err_est_bin_range_lo,
		multi_gpu=using_multi_gpu)

	print('saving best model')

	FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
	torch.save(best_model.state_dict(), FINAL_MODEL_FILE)

	print('fun fun in the sun, the training is done :)')
