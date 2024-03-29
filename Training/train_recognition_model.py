import copy
import time

import torch
from tqdm import tqdm


def train_recognition_model(
		model,
		criterion,
		optimizer,
		scheduler,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		num_epochs=25,
		multi_gpu=False
):
	since = time.time()

	images = next(iter(data_loaders['train']))['image'].to(device)
	writer.add_graph(model, images)
	# writer.close()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		if epoch == 15:
			if multi_gpu:
				model.module.freeze_base_cnn(False)
			else:
				model.freeze_base_cnn(False)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for i, batch in enumerate(tqdm(data_loaders[phase])):
				inputs = batch['image'].to(device)
				labels = batch['label'].to(device).long()

				# zero the parameter gradients
				optimizer.zero_grad()

				
				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					if phase == 'train':
						# note: this is for arc margin classifier
						#outputs = model(inputs, labels, device)
						# note: this is for base recog classifier
						outputs = model(inputs, device)
					else:
						outputs = model(inputs)
					# outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)

				classification_offset = torch.abs(preds - labels.data)

				running_corrects += torch.sum(classification_offset == 0)

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
			writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model
