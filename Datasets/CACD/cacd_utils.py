import glob
import sys
import mat73
import pickle
import json
from PIL import Image, ImageChops
import numpy as np
import h5py

"""
CACD data format
-----------------

After some research:
- the MAT file doesn't include the image data itself; rather metadata (labels) + feature vector 
(doesn't seem to be usable)
- cacd.pickle is a pickle copy of the dictionary in the mat file
- seems that the file names are of the following format:
<age>_<name with '_' between words>_<some number>.jpg
this is based on some example I've seen (in the metadata dictionary there's age metadata)
"""


def get_metadata(filepath, verify=True, os_type="unix"):
	filepath_jpg_split = filepath.split(".jpg")
	if verify:
		if len(filepath_jpg_split[-1]) != 0:
			print(filepath)
			print("bad path : not a jpg file")
			sys.exit(1)
	if os_type == "win":
		file_name_no_prefix = filepath_jpg_split[0].split("\\")[-1]
	elif os_type == "unix":
		file_name_no_prefix = filepath_jpg_split[0].split("/")[-1]
	else:
		print("unsupported os_type")
		sys.exit(1)

	split_path = file_name_no_prefix.split('_')
	if verify:
		if not split_path[0].isdigit():
			print(filepath)
			print("file name doesn't start with a number")
			sys.exit(1)

	subject_id = ''.join(split_path[1:-1])
	age = int(split_path[0])

	return {
		"id_num" : subject_id,
		"age" : age
	}

def get_subject_ids(dataset_path, os_type="unix"):
	image_files = glob.glob(dataset_path + "/*.jpg")
	subject_ids = list(np.unique([get_metadata(filepath, True, os_type)["id_num"] for filepath in image_files]))
	return subject_ids

def train_test_split_subject_ids(dataset_path, os_type="unix", train_set_factor=0.8):
	subject_ids = get_subject_ids(dataset_path, os_type)
	subject_ids_train_set = list(np.random.choice(subject_ids, int(len(subject_ids)*train_set_factor), replace=False))
	subject_ids_test_set = [subject_id for subject_id in subject_ids if subject_id not in subject_ids_train_set]
	return subject_ids_train_set, subject_ids_test_set

def train_set_split(dataset_path, os_type="unix", train_set_factor=0.8, limit=-1):
	subject_ids_train_set, subject_ids_test_set = train_test_split_subject_ids(dataset_path, os_type, train_set_factor)
	image_files = glob.glob(dataset_path + "/*.jpg")
	x_train = list()
	y_train = list()
	x_test = list()
	y_test = list()
	for i, im_file in enumerate(image_files):
		if i == limit:
			break
		if i % 5000 == 0:
			print(f"progress: {100*i/len(image_files)}%")
		metadata = get_metadata(im_file, True, os_type)
		sb_id = metadata["id_num"]
		# age = metadata["age"]
		metadata_json = json.dumps(metadata).encode('utf-8')
		image = Image.open(im_file)
		image_arr = np.array(image)
		if sb_id in subject_ids_train_set:
			x_train.append(image_arr)
			y_train.append(metadata_json)
		elif sb_id in subject_ids_test_set:
			x_test.append(image_arr)
			y_test.append(metadata_json)
		else:
			print("error")
			sys.exit(1)
	return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test) 

def verify_same_content():
	a = mat73.loadmat('F:/age_estimation_with_error_estimator/Datasets/CACD_data_mat/celebrity2000.mat')
		
	with open('F:/age_estsimation_with_error_estimator/Datasets/CACD/cacd.pickle', 'rb') as handle:
		b = pickle.load(handle)

	print(a == b)

# unit tests
if __name__ == "__main__":		
	#subject_ids = get_subject_ids("F:/age_estimation_with_error_estimator/Datasets/CACD_data/CACD2000/CACD2000", os_type="win")
	
	x_train, y_train, x_test, y_test = train_set_split("F:/age_estimation_with_error_estimator/Datasets/CACD_data/CACD2000/CACD2000", os_type="win", train_set_factor=0.8, limit=-1)


	# Open HDF5 file and write in the data_dict structure and info
	f = h5py.File('se_data/se_dataset_cacd.hdf5', 'w')
	f.create_dataset('train_img', data=x_train)
	f.create_dataset('train_labels', data=y_train)
	f.create_dataset('test_img', data=x_test)
	f.create_dataset('test_labels', data=y_test)
	f.close()
	