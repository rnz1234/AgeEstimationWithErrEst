import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from Datasets.AFAD.AFADClassifierDataset import AFADClassifierDataset
from Models.AgeClassifier import AgeClassifier


age_interval = 5
min_age = 15
max_age = 40
BATCH_SIZE = 128

NumLabels = int(max_age / age_interval - min_age / age_interval + 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

model = AgeClassifier(NumLabels)
model.to(device)
model_path = 'weights/AFAD/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_256'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
model.load_state_dict(torch.load(FINAL_MODEL_FILE))
model.eval()
for param in model.parameters():
	param.requires_grad = False


test_ds = AFADClassifierDataset(
	'./Datasets/AFAD/aligned_data/afad_test.h5',
	min_age=min_age,
	max_age=max_age,
	age_interval=age_interval,
	transform=transforms.Compose([
		transforms.ToTensor()
	])
)

data_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

offsets = []
for i, batch in enumerate(tqdm(data_loader)):
	inputs = batch['image'].to(device)
	labels = batch['label'].to(device).long()

	outputs = model(inputs)
	_, preds = torch.max(outputs, 1)

	classification_offset = torch.abs(preds - labels.data)

	offsets.extend(classification_offset)

offsets = torch.stack(offsets, dim=0)

AllErrors = 100 * (offsets > 0).sum() / float(offsets.shape[0])
P1Error = 100 * (offsets == 1).sum() / float(offsets.shape[0])
P2Error = 100 * (offsets == 2).sum() / float(offsets.shape[0])
P3Error = 100 * (offsets == 3).sum() / float(offsets.shape[0])
AverageLoss = (age_interval * P1Error + 2 * age_interval * P2Error + 3 * age_interval * P3Error) / 100

print('AllErrors: ' + AllErrors)
print('P1Error: ' + P1Error)
print('P2Error: ' + P2Error)
print('P3Error: ' + P3Error)
print('AverageLoss: ' + AverageLoss)


