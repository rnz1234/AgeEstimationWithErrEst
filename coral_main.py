# this was adapted from  https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb
# todo: this is without the class imporatnce part, found in the project github

import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F

from Datasets.Morph2.DataParser import DataParser
from Datasets.Morph2.Morph2CoralDataset import Morph2CoralDataset
from Models.Resnet34 import resnet34
from Training.train_coral import train_coral_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

min_age = 15
max_age = 80
NUM_CLASSES = max_age - min_age + 1
BATCH_SIZE = 512
NUM_EPOCHS = 150
LEARNING_RATE = 0.0005
RANDOM_SEED = 123
GRAYSCALE = False

# Load data
data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
data_parser.initialize_data()


train_ds = Morph2CoralDataset(
	data_parser.x_train,
	data_parser.y_train,
	NUM_CLASSES,
	transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.RandomCrop((120, 120)),
		transforms.ToTensor()
	])
)

test_ds = Morph2CoralDataset(
	data_parser.x_test,
	data_parser.y_test,
	NUM_CLASSES,
	transform=transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.CenterCrop((120, 120)),
		transforms.ToTensor()
	])
)

image_datasets = {
	'train': train_ds,
	'val': test_ds
}

data_loaders = {
	x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


# create model and parameters
def cost_fn(logits, levels):
	val = (-torch.sum((F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels)), dim=1))
	return torch.mean(val)


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = resnet34(NUM_CLASSES, GRAYSCALE)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

### Train ###
writer = SummaryWriter('logs/Morph2/Coral/Adam_lr_5e4')

best_classification_model = train_coral_model(
	model,
	cost_fn,
	optimizer,
	data_loaders,
	dataset_sizes,
	device,
	writer,
	num_epochs=NUM_EPOCHS
)

print('saving best model')

model_path = 'weights/Morph2/Coral/Adam_lr_5e4'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
torch.save(best_classification_model.state_dict(), FINAL_MODEL_FILE)

print('exiting')




