import h5py
import numpy as np



class DataParser:
	def __init__(self, hdf5_file_path):
		self.hdf5_file_path = hdf5_file_path
		self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

	def initialize_data(self):
		self.x_train, self.y_train, self.x_test, self.y_test = self.read_data_from_h5_file(self.hdf5_file_path)

	@staticmethod
	def read_data_from_h5_file(hdf5_path):
		hdf5_file = h5py.File(hdf5_path, "r")

		return hdf5_file["train_img"][:], hdf5_file["train_labels"][:], \
		       hdf5_file["test_img"][:], hdf5_file["test_labels"][:]




class DataParserWithLimit:
	def __init__(self, hdf5_file_path, small_data=False):
		self.hdf5_file_path = hdf5_file_path
		self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
		self.small_data = small_data

	def initialize_data(self):
		self.x_train, self.y_train, self.x_test, self.y_test = self.read_data_from_h5_file(self.hdf5_file_path, self.small_data)

	@staticmethod
	def read_data_from_h5_file(hdf5_path, small_data):
		hdf5_file = h5py.File(hdf5_path, "r")

		# import pdb
		# pdb.set_trace()

		if small_data:
			total_idxs_train = np.arange(len(hdf5_file["train_img"]))
			chosen_idxs_train = np.sort(np.random.choice(total_idxs_train, 100, replace=False))
			total_idxs_test = np.arange(len(hdf5_file["test_img"]))
			chosen_idxs_test = np.sort(np.random.choice(total_idxs_test, 100, replace=False))
			return hdf5_file["train_img"][chosen_idxs_train], hdf5_file["train_labels"][chosen_idxs_train], \
		       hdf5_file["test_img"][chosen_idxs_test], hdf5_file["test_labels"][chosen_idxs_test]
			# return hdf5_file["train_img"][:4000], hdf5_file["train_labels"][:4000], \
		    #    hdf5_file["test_img"][:1000], hdf5_file["test_labels"][:1000]
		else:
			return hdf5_file["train_img"][:], hdf5_file["train_labels"][:], \
				hdf5_file["test_img"][:], hdf5_file["test_labels"][:]

		