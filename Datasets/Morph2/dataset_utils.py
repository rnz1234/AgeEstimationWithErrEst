import json
import numpy as np

# getting id map to indexes list
def get_id2idxs(y_data_set):
	id2idxs = dict()
	for i in range(len(y_data_set)):
		cur_id = json.loads(y_data_set[i])["id_num"]
		if cur_id not in id2idxs.keys():
			id2idxs[cur_id] = [i]
		else:
			id2idxs[cur_id].append(i)
	return id2idxs

# ids split to non overlapping sets
def split_ids_to_2_sets(id2idxs, set_1_factor=0.05):
	total_ids = list(id2idxs.keys())
	set_1_ids = list(np.random.choice(total_ids, int(len(total_ids)*set_1_factor), replace=False))
	set_2_ids = [id_num for id_num in total_ids if id_num not in set_1_ids]
	return set_1_ids, set_2_ids

# getting the list of indexes per set
def get_idxs_list_from_ids(set_ids, id2idxs):
	set_idxs = []
	for id_num in set_ids:
		set_idxs = set_idxs + id2idxs[id_num]
	return set_idxs

# get id isolated indexes split
def split_set_to_id_isolated_idxs_sets(y_data_set, set_1_factor=0.05):
	id2idxs = get_id2idxs(y_data_set)
	set_1_ids, set_2_ids = split_ids_to_2_sets(id2idxs, set_1_factor)
	set_1_idxs = get_idxs_list_from_ids(set_1_ids, id2idxs)
	set_2_idxs = get_idxs_list_from_ids(set_2_ids, id2idxs)
	return set_1_idxs, set_2_idxs

# generate dist and isol_test sets
def gen_dist_and_isol_test_sets(x_test, y_test, im2age_map_test_orig, dist_indexes, isolated_test_indexed):
	# generate sets
	y_test_dist = y_test[dist_indexes]
	x_test_dist = x_test[dist_indexes]
	y_test_isol = y_test[isolated_test_indexed]
	x_test_isol = x_test[isolated_test_indexed]
	# generate maps
	im2age_map_dist = dict()
	for new_idx, idx in enumerate(dist_indexes):
		im2age_map_dist[str(new_idx)] = im2age_map_test_orig[str(idx)]
	im2age_map_isol = dict()
	for new_idx, idx in enumerate(isolated_test_indexed):
		im2age_map_isol[str(new_idx)] = im2age_map_test_orig[str(idx)]

	return x_test_dist, y_test_dist, im2age_map_dist, \
		   x_test_isol, y_test_isol, im2age_map_isol

# split to dist and isol_test sets
def split_to_dist_and_isol_test_sets(x_test, y_test, im2age_map_test_orig, dist_set_size_factor=0.05):
	#  split indexes randomly
	dist_indexes, isolated_test_indexed = split_set_to_id_isolated_idxs_sets(y_test, set_1_factor=dist_set_size_factor)
	# generate sets
	return gen_dist_and_isol_test_sets(x_test, y_test, im2age_map_test_orig, dist_indexes, isolated_test_indexed)