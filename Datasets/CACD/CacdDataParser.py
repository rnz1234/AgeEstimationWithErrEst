import h5py
import numpy as np
from Datasets.CACD.cacd_utils import train_set_split

class CacdDataParser:
	def __init__(self, dataset_path):
		self.dataset_path = dataset_path
		self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

	def initialize_data(self):
		self.x_train, self.y_train, self.x_test, self.y_test = train_set_split(self.dataset_path, os_type="win", train_set_factor=0.8, limit=1000)

	# def initialize_data(self):
	# 	self.x_train, self.y_train, self.x_test, self.y_test = self.read_data_from_h5_file(self.hdf5_file_path)

	# @staticmethod
	# def read_data_from_h5_file(hdf5_path):
	# 	hdf5_file = h5py.File(hdf5_path, "r")

	# 	return hdf5_file["train_img"][:], hdf5_file["train_labels"][:], \
	# 	       hdf5_file["test_img"][:], hdf5_file["test_labels"][:]