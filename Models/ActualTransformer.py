import torch
import torch.nn.functional as F
from torch import nn
from Models.keller_transformer import Transformer

class ActualTransformer(nn.Module):
	def __init__(self, backbone, backbone_feat_size, num_classes, min_age, age_interval, hidden_dim=256, nheads=8,
	             num_encoder_layers=6, num_decoder_layers=6):
		super().__init__()
		self.labels = range(num_classes)

		# backbone
		self.backbone = backbone

		# create conversion layer
		self.dimensionality_reduction_layer = nn.Linear(backbone_feat_size, hidden_dim)

		# create a default PyTorch transformer
		# self.transformer = nn.Transformer(
		# 	hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

		self.transformer = Transformer(
			d_model=hidden_dim, dropout=0.5, nhead=nheads,
			dim_feedforward=hidden_dim,
			num_encoder_layers=num_encoder_layers,
			num_decoder_layers=num_decoder_layers,
			normalize_before=False, return_intermediate_dec=False)

		# prediction heads
		self.class_head = nn.Linear(hidden_dim, num_classes)

		self.regression_heads = []
		self.centers = []
		for i in self.labels:
			self.regression_heads.append(nn.Linear(hidden_dim, 1))
			center = min_age + 0.5 * age_interval + i * age_interval
			self.centers.append(center)

		self.regression_heads = nn.ModuleList(self.regression_heads)

		# output positional encodings (object queries)
		#TODO: maybe 2 queries? one for each task
		self.query_pos = nn.Parameter(torch.rand(2, hidden_dim), requires_grad=True)

		# spatial positional encodings
		# note that in baseline DETR we use sine positional encodings
		# self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2), requires_grad=True)
		# self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2), requires_grad=True)

	def forward(self, input):
		unpacked_input = input.view(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4])

		x = self.backbone(unpacked_input)
		# x = F.leaky_relu(x)

		h = self.dimensionality_reduction_layer(x)
		h = F.leaky_relu(h)
		h = F.normalize(h, dim=1, p=2)

		# h = h.view(input.shape[1], input.shape[0], -1)
		h = h.reshape((int(h.shape[0] / input.shape[1]), int(input.shape[1]), h.shape[1]))

		# construct positional encodings
		# H, W = h.shape[-2:]
		# pos = torch.cat([
		# 	self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
		# 	self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
		# ], dim=-1).flatten(0, 1).unsqueeze(1)

		# propagate through the transformer
		h = torch.transpose(h, 0, 1)
		query_embed = torch.cat(h.shape[1] * [self.query_pos])

		h, memory = self.transformer(
			# pos + 0.1 * h.flatten(2).permute(2, 0, 1),
			src=h,
			mask=None,
			query_embed=query_embed,
			pos_embed=None
		)
		h = h.transpose(1, 0).squeeze()

		splitted = torch.split(h, h.shape[0], dim=1)

		# finally project transformer outputs to class labels and bounding boxes
		base_embedding_first_stage = splitted[0].mean(axis=1)
		base_embedding_second_stage = splitted[1].mean(axis=1)

		# base_embedding = h.mean(axis=1)

		classification_logits = self.class_head(base_embedding_first_stage)
		weights = nn.Softmax()(classification_logits)

		t = []
		for i in self.labels:
			t.append(torch.squeeze(self.regression_heads[i](base_embedding_second_stage) + self.centers[i]) * weights[:, i])

		age_pred = torch.stack(t, dim=0).sum(dim=0) / torch.sum(weights, dim=1)

		return classification_logits, age_pred
