import copy
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_keller_ordinal(
		model,
		WeightedBceLoss,
		optimizer,
		scheduler,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		NumLabels=1,
		min_age=15,
		num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		if epoch == 5:
			model.FreezeBaseCnn(False)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_mae = 0.0

			for i, batch in enumerate(tqdm(data_loaders[phase])):
				inputs = batch['image'].to(device)
				classification_labels = batch['classification_label'].to(device).float()
				ages = batch['age'].to(device).float()

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					result = model(inputs)
					# _, class_preds = torch.max(classification_logits, 1)

					# reg_loss = criterion_reg(age_pred, ages)
					# cls_loss = criterion_cls(classification_logits, classification_labels.long())
					# mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
					# loss = reg_loss + cls_loss + mean_loss + var_loss

					loss = WeightedBceLoss(result['OrdinalClass'], ages.round().long() - min_age, compute_weights=True)

					if phase == 'train':
						loss.backward()
						optimizer.step()

				age_pred = (result['OrdinalClass'] > 0).sum(1) + min_age

				running_loss += loss.item() * inputs.size(0)
				running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_mae = running_mae / dataset_sizes[phase]

			if phase == 'train':
				scheduler.step(epoch_loss)

			writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
			writer.add_scalar('Mae/{}'.format(phase), epoch_mae, epoch)

			print('{} Loss: {:.4f} mae: {:.4f}'.format(phase, epoch_loss, epoch_mae))

			# deep copy the model
			if phase == 'val' and epoch_mae < best_mae:
				best_mae = epoch_mae
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model
