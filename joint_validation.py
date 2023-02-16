import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm

from Datasets.AFAD.AFADRegressorDataset import AFADRegressorDataset
from Models.AgeClassifier import AgeClassifier
from Models.AgeMultiHeadRegressor import AgeMultiHeadRegressor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

age_interval = 5
min_age = 15
max_age = 40
BATCH_SIZE = 128

num_labels = int(max_age / age_interval - min_age / age_interval + 1)

classification_model_path = \
	'weights/AFAD/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_256/weights.pt'
classification_model = AgeClassifier(num_labels)
classification_model.to(device)
classification_model.load_state_dict(torch.load(classification_model_path))
for param in classification_model.parameters():
	param.requires_grad = False
classification_model.eval()

regression_model_path = \
	'weights/AFAD/multihead_regression/RangerLars_unfreeze_at_15_lr_1e2_steplr_10_01_256_dropout/weights.pt'
multihead_regrssion_model = AgeMultiHeadRegressor(num_labels, age_interval, min_age, max_age)
multihead_regrssion_model.to(device)
multihead_regrssion_model.load_state_dict(torch.load(regression_model_path))
for param in multihead_regrssion_model.parameters():
	param.requires_grad = False
multihead_regrssion_model.eval()


# Load data
# data_parser = DataParser()
# data_parser.initialize_data()
#
# test_ds = Morph2RegressorDataset(
# 	data_parser.x_test,
# 	data_parser.y_test,
# 	min_age,
# 	age_intareval,
# 	num_labels,
# 	transform=transforms.Compose([
# 		transforms.ToTensor()
# 	])
# )

test_ds = AFADRegressorDataset(
	'./Datasets/AFAD/aligned_data/afad_test.h5',
	min_age=min_age,
	max_age=max_age,
	age_interval=age_interval,
	transform=transforms.Compose([
		transforms.ToTensor()
	])
)

data_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataset_sizes = len(test_ds)

age_preds = []
age_labels = []
for i, batch in enumerate(tqdm(data_loader)):
	inputs = batch['image'].to(device)
	labels = batch['label'].to(device).float()
	# weights = batch['weights'].to(device)

	logits = classification_model(inputs)
	weights = nn.Softmax()(logits)

	# classes = torch.argmax(weights, 1)
	# one_hot = torch.zeros(weights.size(), device=device)
	# one_hot.scatter_(1, classes.view(-1, 1).long(), 1)

	torch.cuda.empty_cache()

	preds = multihead_regrssion_model(inputs, weights)

	age_preds.extend(preds)
	age_labels.extend(labels)

mae = torch.nn.L1Loss()(torch.stack(age_preds, dim=0), torch.stack(age_labels, dim=0))
print(mae)


print('exiting...')
