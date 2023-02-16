import copy
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

def validate(model, val_data_loader, iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_size, writer, use_cls_mean_var):
	print('running on validation set...')

	model.eval()

	running_loss = 0.0
	running_mae = 0.0
	running_corrects = 0.0
	running_p1_error = 0.0
	running_p2_and_above_error = 0.0

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		classification_labels = batch['classification_label'].to(device).float()
		ages = batch['age'].to(device).float()

		with torch.no_grad():
			classification_logits, age_pred = model(inputs)
			_, class_preds = torch.max(classification_logits, 1)

		reg_loss = criterion_reg(age_pred, ages)
		loss = reg_loss
		if use_cls_mean_var:
			cls_loss = criterion_cls(classification_logits, classification_labels.long())
			mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
			loss += cls_loss + mean_loss + var_loss

		running_loss += loss.item() * inputs.size(0)
		running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

		if use_cls_mean_var:
			classification_offset = torch.abs(class_preds - classification_labels.data)
			running_corrects += torch.sum(classification_offset == 0)
			running_p1_error += torch.sum(classification_offset == 1)
			running_p2_and_above_error += torch.sum(classification_offset >= 2)

	epoch_loss = running_loss / dataset_size
	epoch_mae = running_mae / dataset_size

	if use_cls_mean_var:
		epoch_acc = running_corrects.double() / dataset_size
		epoch_p1_error = running_p1_error.double() / dataset_size
		epoch_p2_and_above_error = running_p2_and_above_error.double() / dataset_size

	writer.add_scalar('Loss/val', epoch_loss, iter)
	writer.add_scalar('Mae/val', epoch_mae, iter)

	if use_cls_mean_var:
		writer.add_scalar('Accuracy/val', epoch_acc, iter)
		writer.add_scalar('Accuracy_+-1/val', epoch_p1_error, iter)
		writer.add_scalar('Accuracy_+-2_and_above/val', epoch_p2_and_above_error, iter)

	# writer.add_scalar('alpha', model.alpha.cpu().detach().numpy().squeeze(), iter)

	print('{} Loss: {:.4f} mae: {:.4f}'.format('val', epoch_loss, epoch_mae))

	return epoch_mae


def train_unified_model_iter(
		model,
		criterion_reg,
		criterion_cls,
		mean_var_criterion,
		optimizer,
		scheduler,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		model_path,
		NumLabels=1,
		num_epochs=25,
		validate_at_k=100,
		use_cls_mean_var=True
):

	since = time.time()

	scaler = GradScaler()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	running_loss = 0.0
	running_mae = 0.0
	running_corrects = 0.0
	running_p1_error = 0.0
	running_p2_and_above_error = 0.0

	iter = 0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# if epoch == 15:
		# 	model.FreezeBaseCnn(False)

		phase = 'train'
		for batch in tqdm(data_loaders[phase]):
			if iter % validate_at_k == 0 and iter != 0:
				norm = validate_at_k * inputs.size(0)

				epoch_loss = running_loss / norm
				epoch_mae = running_mae / norm

				if use_cls_mean_var:
					epoch_acc = running_corrects.double() / norm
					epoch_p1_error = running_p1_error.double() / norm
					epoch_p2_and_above_error = running_p2_and_above_error.double() / norm

				writer.add_scalar('Loss/{}'.format(phase), epoch_loss, iter)
				writer.add_scalar('Mae/{}'.format(phase), epoch_mae, iter)

				if use_cls_mean_var:
					writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, iter)
					writer.add_scalar('Accuracy_+-1/{}'.format(phase), epoch_p1_error, iter)
					writer.add_scalar('Accuracy_+-2_and_above/{}'.format(phase), epoch_p2_and_above_error, iter)

				print('{} Loss: {:.4f} mae: {:.4f}'.format(phase, epoch_loss, epoch_mae))

				val_mae = validate(model, data_loaders['val'], iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_sizes['val'], writer, use_cls_mean_var=use_cls_mean_var)

				# deep copy the model
				if val_mae < best_mae:
					best_mae = val_mae
					best_model_wts = copy.deepcopy(model.state_dict())

					FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae))
					torch.save(best_model_wts, FINAL_MODEL_FILE)

				model.train()

				running_loss = 0.0
				running_mae = 0.0
				running_corrects = 0.0
				running_p1_error = 0.0
				running_p2_and_above_error = 0.0

			iter += 1

			inputs = batch['image'].to(device)
			classification_labels = batch['classification_label'].to(device).float()
			ages = batch['age'].to(device).float()

			optimizer.zero_grad()

			with autocast():
				classification_logits, age_pred = model(inputs)
				_, class_preds = torch.max(classification_logits, 1)

				reg_loss = criterion_reg(age_pred, ages)
				loss = reg_loss
				if use_cls_mean_var:
					cls_loss = criterion_cls(classification_logits, classification_labels.long())
					mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
					loss += cls_loss + mean_loss + var_loss

			# loss.backward()
			# optimizer.step()

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			running_loss += loss.item() * inputs.size(0)
			running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

			if use_cls_mean_var:
				classification_offset = torch.abs(class_preds - classification_labels.data)
				running_corrects += torch.sum(classification_offset == 0)
				running_p1_error += torch.sum(classification_offset == 1)
				running_p2_and_above_error += torch.sum(classification_offset >= 2)

			# scheduler.step(epoch_mae)
			scheduler.step()


		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model










def validate_with_err_est(model, val_data_loader, iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_size, writer, use_cls_mean_var, criterion_err_est, err_est_bin_range_lo, epoch):
	print('running on validation set...')

	model.eval()

	running_loss = 0.0
	running_mae = 0.0
	running_corrects = 0.0
	running_p1_error = 0.0
	running_p2_and_above_error = 0.0

	running_loss_err_est = 0.0
	running_corrects_err_est = 0.0

	#for i, batch in enumerate(tqdm(val_data_loader)):
	for batch in tqdm(val_data_loader, position=0, leave=True):
		inputs = batch['image'].to(device)
		classification_labels = batch['classification_label'].to(device).float()
		ages = batch['age'].to(device).float()

		with torch.no_grad():
			classification_logits, age_pred, age_pred_err_est_head = model(inputs)
			_, class_preds = torch.max(classification_logits, 1)
			pred_err_prob = torch.sigmoid(age_pred_err_est_head)
			pred_err = torch.round(pred_err_prob)

		# import pdb
		# pdb.set_trace()

		label_above_lo = (torch.abs(age_pred-ages) > err_est_bin_range_lo).to(dtype=torch.float64)
		age_pred_err_est_head_flat = age_pred_err_est_head.view(-1)
		loss_err_est = criterion_err_est(age_pred_err_est_head_flat, label_above_lo)

		reg_loss = criterion_reg(age_pred, ages)
		loss = reg_loss + loss_err_est
		if use_cls_mean_var:
			cls_loss = criterion_cls(classification_logits, classification_labels.long())
			mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
			loss += cls_loss + mean_loss + var_loss + loss_err_est

		running_loss += loss.item() * inputs.size(0)
		running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

		running_loss_err_est += loss_err_est.item() * inputs.size(0)
		classification_err_est_offset = torch.abs(pred_err.view(-1) - label_above_lo)
		running_corrects_err_est += torch.sum(classification_err_est_offset == 0)

		if use_cls_mean_var:
			classification_offset = torch.abs(class_preds - classification_labels.data)
			running_corrects += torch.sum(classification_offset == 0)
			running_p1_error += torch.sum(classification_offset == 1)
			running_p2_and_above_error += torch.sum(classification_offset >= 2)

	epoch_loss = running_loss / dataset_size
	epoch_mae = running_mae / dataset_size

	epoch_loss_err_est = running_loss_err_est / dataset_size
	epoch_acc_err_est = running_corrects_err_est.double() / dataset_size

	if use_cls_mean_var:
		epoch_acc = running_corrects.double() / dataset_size
		epoch_p1_error = running_p1_error.double() / dataset_size
		epoch_p2_and_above_error = running_p2_and_above_error.double() / dataset_size

	writer.add_scalar('Loss/val', epoch_loss, iter)
	writer.add_scalar('Mae/val', epoch_mae, iter)

	writer.add_scalar('LossErrEst/val', epoch_loss_err_est, iter)
	writer.add_scalar('AccuracyErrEst/val', epoch_acc_err_est, iter)
					

	if use_cls_mean_var:
		writer.add_scalar('Accuracy/val', epoch_acc, iter)
		writer.add_scalar('Accuracy_+-1/val', epoch_p1_error, iter)
		writer.add_scalar('Accuracy_+-2_and_above/val', epoch_p2_and_above_error, iter)

	# writer.add_scalar('alpha', model.alpha.cpu().detach().numpy().squeeze(), iter)

	print('{} Loss (age): {:.4f} mae (age): {:.4f} Loss (err): {:.4f} acc (err): {:.4f}'.format('val', epoch_loss, epoch_mae, epoch_loss_err_est, epoch_acc_err_est))

	print("---------------------------------------")
	print("- Validation - error range detection info:")
	# Save confusion matrix to Tensorboard
	#writer.add_figure("Confusion matrix/val", create_confusion_matrix(val_data_loader, model, device, err_est_bin_range_lo), epoch)
	auc = create_confusion_matrix(val_data_loader, model, device, err_est_bin_range_lo)

	writer.add_scalar('AUC/val', auc, iter)

	return epoch_mae, epoch_acc_err_est





def train_unified_model_with_err_est_iter(
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
		NumLabels=1,
		num_epochs=25,
		validate_at_k=100,
		use_cls_mean_var=True,
		err_est_bin_range_lo=4,
		multi_gpu=False
):

	since = time.time()

	scaler = GradScaler()

	# scaler_err_est = GradScaler()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	running_loss = 0.0
	running_mae = 0.0
	running_corrects = 0.0
	running_p1_error = 0.0
	running_p2_and_above_error = 0.0

	running_loss_err_est = 0.0
	running_corrects_err_est = 0.0
	
	acc_norm = 0.0

	iter = 0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		if epoch == 0: #7
			if multi_gpu:
				model.module.FreezeBaseCnn(False)
			else:
				model.FreezeBaseCnn(False)
			#model.FreezeBaseCnn(False)

		

		phase = 'train'
		for batch in tqdm(data_loaders[phase]):
			if iter % validate_at_k == 0 and iter != 0:
				#norm = validate_at_k * inputs.size(0)
				norm = acc_norm

				# import pdb
				# pdb.set_trace()

				epoch_loss = running_loss / norm
				epoch_mae = running_mae / norm

				epoch_loss_err_est = running_loss_err_est / norm
				epoch_acc_err_est = running_corrects_err_est.double() / norm

				if use_cls_mean_var:
					epoch_acc = running_corrects.double() / norm
					epoch_p1_error = running_p1_error.double() / norm
					epoch_p2_and_above_error = running_p2_and_above_error.double() / norm

				writer.add_scalar('Loss/{}'.format(phase), epoch_loss, iter)
				writer.add_scalar('Mae/{}'.format(phase), epoch_mae, iter)

				writer.add_scalar('LossErrEst/{}'.format(phase), epoch_loss_err_est, iter)
				writer.add_scalar('AccuracyErrEst/{}'.format(phase), epoch_acc_err_est, iter)
					
				if use_cls_mean_var:
					writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, iter)
					writer.add_scalar('Accuracy_+-1/{}'.format(phase), epoch_p1_error, iter)
					writer.add_scalar('Accuracy_+-2_and_above/{}'.format(phase), epoch_p2_and_above_error, iter)

				print('{} Loss (age): {:.4f} mae (age): {:.4f} Loss (err): {:.4f} acc (err): {:.4f}'.format(phase, epoch_loss, epoch_mae, epoch_loss_err_est, epoch_acc_err_est))

				
				val_mae, val_err_est_acc = validate_with_err_est(model, data_loaders['val'], iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_sizes['val'], writer, use_cls_mean_var=use_cls_mean_var, criterion_err_est=criterion_err_est, err_est_bin_range_lo=err_est_bin_range_lo, epoch=epoch)

				# deep copy the model
				if val_mae < best_mae:
					best_mae = val_mae
					best_model_wts = copy.deepcopy(model.state_dict())

					FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae))
					torch.save(best_model_wts, FINAL_MODEL_FILE)

				model.train()

				running_loss = 0.0
				running_mae = 0.0
				running_corrects = 0.0
				running_p1_error = 0.0
				running_p2_and_above_error = 0.0

				running_loss_err_est = 0.0
				running_corrects_err_est = 0.0
				acc_norm = 0.0

			iter += 1

			

			inputs = batch['image'].to(device)
			classification_labels = batch['classification_label'].to(device).float()
			ages = batch['age'].to(device).float()

			acc_norm += inputs.size(0) 

			optimizer.zero_grad()
			#optimizer_err_est.zero_grad()

			with autocast():
				classification_logits, age_pred, age_pred_err_est_head = model(inputs)
				_, class_preds = torch.max(classification_logits, 1)

				pred_err_prob = torch.sigmoid(age_pred_err_est_head)
				pred_err = torch.round(pred_err_prob)

				# import pdb
				# pdb.set_trace()

				label_above_lo = (torch.abs(age_pred-ages) > err_est_bin_range_lo).to(dtype=torch.float64)
				age_pred_err_est_head_flat = age_pred_err_est_head.view(-1)
				loss_err_est = criterion_err_est(age_pred_err_est_head_flat, label_above_lo)

				reg_loss = criterion_reg(age_pred, ages)
				loss = reg_loss + loss_err_est
				if use_cls_mean_var:
					cls_loss = criterion_cls(classification_logits, classification_labels.long())
					mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
					loss += cls_loss + mean_loss + var_loss + loss_err_est

			# loss.backward()
			# optimizer.step()

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			# scaler_err_est.scale(loss_err_est).backward()
			# scaler_err_est.step(optimizer_err_est)
			# scaler_err_est.update()

			running_loss += loss.item() * inputs.size(0)
			running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

			running_loss_err_est += loss_err_est.item() * inputs.size(0)
			classification_err_est_offset = torch.abs(pred_err.view(-1) - label_above_lo)
			running_corrects_err_est += torch.sum(classification_err_est_offset == 0)

			if use_cls_mean_var:
				classification_offset = torch.abs(class_preds - classification_labels.data)
				running_corrects += torch.sum(classification_offset == 0)
				running_p1_error += torch.sum(classification_offset == 1)
				running_p2_and_above_error += torch.sum(classification_offset >= 2)

			# scheduler.step(epoch_mae)
			scheduler.step()
			#scheduler_err_est.step()


		print()
		print("---------------------------------------")
		print("- Training - error range detection info:")
		# Save confusion matrix to Tensorboard
		#writer.add_figure("Confusion matrix/{}".format('train'), create_confusion_matrix(data_loaders['train'], model, device, err_est_bin_range_lo), epoch)
		auc = create_confusion_matrix(data_loaders['train'], model, device, err_est_bin_range_lo)
		writer.add_scalar('AUC/{}'.format(phase), auc, iter)
		model.train()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model





def create_confusion_matrix(loader, model, device, err_est_bin_range_lo):
	model.eval()
	y_pred = [] # save predction
	y_true = [] # save ground truth
	y_prob_of_1 = []

	# iterate over data
	for batch in tqdm(loader, position=0, leave=True):
		inputs = batch['image'].to(device)
		classification_labels = batch['classification_label'].to(device).float()
		ages = batch['age'].to(device).float()

		with torch.no_grad():
			classification_logits, age_pred, age_pred_err_est_head = model(inputs)  # Feed Network

		pred_err_prob = torch.sigmoid(age_pred_err_est_head)
		y_prob_of_1.extend(pred_err_prob.cpu().numpy().reshape(-1))
		pred_err = torch.round(pred_err_prob).cpu().numpy().reshape(-1)

		y_pred.extend(pred_err)  # save prediction

		label_above_lo = (torch.abs(age_pred-ages) > err_est_bin_range_lo).to(dtype=torch.float64).cpu().numpy()
		y_true.extend(label_above_lo)  # save ground truth

	# fpr, tpr, thresholds = roc_curve(testy, yhat)

	# constant for classes
	classes = ["0", "1"]

	# import pdb
	# pdb.set_trace()

	# Build confusion matrix
	cf_matrix = confusion_matrix(y_true, y_pred, labels=range(0,2))
	df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes) # cf_matrix/np.sum(cf_matrix) * 10

	plt.figure(figsize=(12, 7))    
	sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g')#.get_figure()

	plt.tight_layout()
	plt.title("Confusion Matrix")
	plt.xlabel('Predicted Label') #, fontsize=40)
	plt.ylabel('Actual Label') #, fontsize=40)

	plt.plot()
	plt.show()

	plt.figure(figsize=(12, 7))   
	fpr, tpr, thresholds = roc_curve(y_true, y_prob_of_1)

	gmeans = np.sqrt(tpr * (1-fpr))
	# locate the index of the largest g-mean
	ix = np.argmax(gmeans)
	print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
	# plot the roc curve for the model
	plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
	plt.plot(fpr, tpr, marker='.', label='Logistic')
	plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
	plt.xlabel('FPR') #, fontsize=40)
	plt.ylabel('TPR') #, fontsize=40)
	plt.legend()
	plt.title("ROC Curve")
	plt.show()

	auc = roc_auc_score(y_true, y_prob_of_1)
	print("AUC = {auc}".format(auc=auc))
	return auc