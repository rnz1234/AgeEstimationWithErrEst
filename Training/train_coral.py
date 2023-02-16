import copy
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_coral_model(
		model, criterion, optimizer, data_loaders, dataset_sizes,
		device, writer, scheduler=None, num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_mae = 0.0

			for batch_idx, (inputs, targets, levels) in enumerate(tqdm(data_loaders[phase])):
			# for batch_idx, batch in enumerate(tqdm(data_loaders[phase])):

				inputs = inputs.to(device)
				targets = targets.to(device).float()
				levels = levels.to(device)

				# FORWARD AND BACK PROP
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					logits, probas = model(inputs)
					loss = criterion(logits, levels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				predict_levels = probas > 0.5
				predicted_labels = torch.sum(predict_levels, dim=1)

				running_loss += loss.item() * inputs.size(0)
				running_mae += torch.nn.L1Loss()(predicted_labels, targets) * inputs.size(0)

			if phase == 'train' and scheduler is not None:
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_mae = running_mae / dataset_sizes[phase]

			writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
			writer.add_scalar('Mae/{}'.format(phase), epoch_mae, epoch)

			# print('{} Loss: {:.4f}'.format(phase, epoch_loss))
			print('{} Loss: {:.4f} mae: {:.4f}'.format(phase, epoch_loss, epoch_mae))

			# deep copy the model
			if phase == 'val' and epoch_mae < best_mae:
				best_mae = epoch_mae
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model
