import torch
from skimage import transform
import numpy as np


class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
		max_crop_count how many pixels to crop out from each dimension in case output_size is same as image size
	"""

	def __init__(self, output_size, max_crop_count):
		assert isinstance(output_size, (int, tuple))
		assert isinstance(max_crop_count, int)
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

		self.max_crop_count = max_crop_count

	def __call__(self, image):
		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		if h == new_h and w == new_w:
			top = np.random.randint(0, self.max_crop_count)
			left = np.random.randint(0, self.max_crop_count)
			bottom = np.random.randint(0, self.max_crop_count)
			right = np.random.randint(0, self.max_crop_count)

			image = image[top: h - bottom, left: w - right]
			image = transform.resize(image, (new_h, new_w))

		else:
			top = np.random.randint(0, h - new_h)
			left = np.random.randint(0, w - new_w)

			image = image[top: top + new_h, left: left + new_w]

		return image


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, image):
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return torch.from_numpy(image)
