import torch
import torch.nn as nn
from dgl.nn.pytorch import GINConv
# from torch_geometric.nn import GINConv
import torch.nn.functional as F
import numpy as np


class PPI_model(nn.Module):
	def __init__(self, param):
		super(PPI_model, self).__init__()
		self.num_layers = param['GIN_layers']
		self.dropout = nn.Dropout(param['dropout'])
		self.layers = nn.ModuleList()
		
		self.layers.append(GINConv(nn.Sequential(nn.Linear(param['embedding_dim']*2, param['embedding_dim']),
													nn.ReLU(),
													nn.Linear(param['embedding_dim'], param['embedding_dim']),
													nn.ReLU(),
													nn.BatchNorm1d(param['embedding_dim'])),
									aggregator_type='sum',
									learn_eps=True))

		for i in range(self.num_layers-1):
			self.layers.append(GINConv(nn.Sequential(nn.Linear(param['embedding_dim'], param['embedding_dim']),
													nn.ReLU(),
													nn.Linear(param['embedding_dim'], param['embedding_dim']),
													nn.ReLU(),
													nn.BatchNorm1d(param['embedding_dim'])),
									aggregator_type='sum',
									learn_eps=True))
		self.linear = nn.Sequential(nn.Linear(param['embedding_dim'], param['embedding_dim']),
								nn.ReLU(),
								nn.Dropout(param['dropout']),
								nn.Linear(param['embedding_dim'], param['embedding_dim']))
		self.fc = nn.Sequential(
			nn.Linear(param['embedding_dim'], param['embedding_dim']),
								nn.ReLU(),
								nn.Dropout(param['dropout']),
								nn.Linear(param['embedding_dim'], param['output_dim']))

	def forward(self,g,x,ppi_list,idx):
		# layers = []
		# layers.append(x)
		for l, layer in enumerate(self.layers):
			x = layer(g, x)
			# x = torch.add(x, layers[0])
			x = self.dropout(x)
		x = F.relu(self.linear(x))
		node_id = np.array(ppi_list)[idx]
		x1 = x[node_id[:, 0]]
		x2 = x[node_id[:, 1]]


		x = self.fc(torch.mul(x1, x2))

		return x