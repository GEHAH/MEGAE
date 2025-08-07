import dgl
import sys
import esm
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn.functional as F
sys.path.append('/data2/csz/PPI_tnnls/Models')
from model_utils import InfoNCELoss
from dgl.nn.pytorch import GATConv,HeteroGraphConv,GCN2Conv,GraphConv


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class GNNEncoders(nn.Module):
    def __init__(self, args):
        super(GNNEncoders, self).__init__()
        self.num_layers = args['num_layers']
        self.hidden_dim = args['hidden_dim']
        self.input_dim = args['input_dim']

        self.layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(HeteroGraphConv({
            'residue': GATConv(self.input_dim,self.hidden_dim,num_heads=args['heads'],feat_drop=0.2,residual=True),
            'seq': GATConv(self.input_dim,self.hidden_dim,num_heads=args['heads'],feat_drop=0.2,residual=True),
            'knn': GATConv(self.input_dim,self.hidden_dim,num_heads=args['heads'],feat_drop=0.2,residual=True),
        },aggregate='sum'))
        self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.norms.append(nn.BatchNorm1d(self.input_dim))
        for i in range(self.num_layers - 1):
            self.layers.append(HeteroGraphConv({
            'residue': GATConv(self.hidden_dim,self.hidden_dim,num_heads=args['heads'],feat_drop=0.2,residual=True),
            'seq': GATConv(self.hidden_dim,self.hidden_dim,num_heads=args['heads'],feat_drop=0.2,residual=True),
            'knn': GATConv(self.hidden_dim,self.hidden_dim,num_heads=args['heads'],feat_drop=0.2,residual=True),
            },aggregate='sum'))
            self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

    def forward(self, batch_graph, x):
        for i in range(self.num_layers):
            x = self.norms[i](x)
            x = self.layers[i](batch_graph,{'amino_acid':x})
            x = torch.mean(x['amino_acid'], dim=1)
            x = F.relu(self.fcs[i](x))
            if i < self.num_layers - 1:
                x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def encoder(self,batch_graph,vq):
        x = batch_graph.ndata['x']
        h = self.forward(batch_graph, x)
        z,_,index = vq(h)
        embed_z = z.detach().cpu()
        batch_graph.ndata['z'] = z
        embed_mean = dgl.mean_nodes(batch_graph, 'z').detach().cpu()
        return embed_z, embed_mean, index

class GNNDecoders(nn.Module):
    def __init__(self, args):
        super(GNNDecoders, self).__init__()
        self.num_layers = args['num_layers']
        self.hidden_dim = args['hidden_dim']
        self.input_dim = args['input_dim']
        self.layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.layers.append(HeteroGraphConv({
                'residue': GATConv(self.hidden_dim, self.hidden_dim, num_heads=args['heads'], feat_drop=0.2, residual=True),
                'seq': GATConv(self.hidden_dim, self.hidden_dim, num_heads=args['heads'], feat_drop=0.2, residual=True),
                'knn': GATConv(self.hidden_dim, self.hidden_dim, num_heads=args['heads'], feat_drop=0.2, residual=True),
            }, aggregate='sum'))
            self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        self.layers.append(HeteroGraphConv({
            'residue': GATConv(self.hidden_dim, self.input_dim, num_heads=args['heads'], feat_drop=0.2, residual=True),
            'seq': GATConv(self.hidden_dim, self.input_dim, num_heads=args['heads'], feat_drop=0.2, residual=True),
            'knn': GATConv(self.hidden_dim, self.input_dim, num_heads=args['heads'], feat_drop=0.2, residual=True),
        }, aggregate='sum'))

    def forward(self, batch_graph, x):
        for i in range(self.num_layers):
            x = self.layers[i](batch_graph, {'amino_acid': x})
            x = torch.mean(x['amino_acid'], dim=1)
            if i < self.num_layers - 1:
                x = self.norms[i](F.relu(self.fcs[i](x)))
                x = F.dropout(x, p=0.2, training=self.training)
        return x

class GNNEncoder(nn.Module):
    def __init__(self, args):
        super(GNNEncoder, self).__init__()
        self.num_layers = args['num_layers']
        self.hidden_dim = args['hidden_dim']
        self.input_dim = args['input_dim']

        self.layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(HeteroGraphConv({
            'residue': GraphConv(self.input_dim,self.hidden_dim),
            'seq': GraphConv(self.input_dim,self.hidden_dim),
            'knn': GraphConv(self.input_dim,self.hidden_dim),
        },aggregate='sum'))
        self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        for i in range(self.num_layers - 1):
            self.layers.append(HeteroGraphConv({
            'residue': GraphConv(self.hidden_dim,self.hidden_dim),
            'seq': GraphConv(self.hidden_dim,self.hidden_dim),
            'knn': GraphConv(self.hidden_dim,self.hidden_dim),
            },aggregate='sum'))
            self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

    def forward(self, batch_graph, x):
        for i in range(self.num_layers):
            # x = self.norms[i](x)
            x = self.layers[i](batch_graph,{'amino_acid':x})
            x = self.norms[i](F.relu(self.fcs[i](x['amino_acid'])))
            # x = F.relu(self.fcs[i](x['amino_acid']))
            if i < self.num_layers - 1:
                x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def encoder(self,batch_graph,vq):
        x = batch_graph.ndata['x']
        h = self.forward(batch_graph, x)
        z,_,index = vq(h)
        # embed_z = z.detach().cpu()
        # batch_graph.ndata['z'] = z
        # embed_mean = dgl.mean_nodes(batch_graph, 'z').detach().cpu()
        embed_z = torch.cat([h,z], dim=-1).detach().cpu()
        batch_graph.ndata['h'] = torch.cat([h,z], dim=-1)
        embed_mean = dgl.mean_nodes(batch_graph, 'h').detach().cpu()

        return embed_z, embed_mean, index

class GNNDecoder(nn.Module):
    def __init__(self, args):
        super(GNNDecoder, self).__init__()
        self.num_layers = args['num_layers']
        self.hidden_dim = args['hidden_dim']
        self.input_dim = args['input_dim']
        self.layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.layers.append(HeteroGraphConv({
            'residue': GraphConv(self.hidden_dim,self.hidden_dim),
            'seq': GraphConv(self.hidden_dim,self.hidden_dim),
            'knn': GraphConv(self.hidden_dim,self.hidden_dim),
                    },aggregate='sum'))
            self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        self.layers.append(HeteroGraphConv({
            'residue': GraphConv(self.hidden_dim,self.hidden_dim),
            'seq': GraphConv(self.hidden_dim,self.hidden_dim),
            'knn': GraphConv(self.hidden_dim,self.hidden_dim),
        },aggregate='sum'))
        self.fcs.append(nn.Linear(self.hidden_dim, self.input_dim))
        self.norms.append(nn.BatchNorm1d(self.hidden_dim))

    def forward(self, batch_graph, x):
        for i in range(self.num_layers):
            x = self.layers[i](batch_graph, {'amino_acid': x})
            x = self.fcs[i](x['amino_acid'])
            if i != self.num_layers-1:
                x = self.norms[i](F.relu(x))
        return x



class esm_encoder(nn.Module):
    def __init__(self,args):
        super(esm_encoder,self).__init__()
        self.declayers = args['declayers']
        self.lora_r = args['lora_r']
        self.lora_alpha = args['lora_alpha']
        self.esm_dim = args['esm_dim']
        self.esm2, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.num_layer = self.esm2.num_layers
        for p in self.esm2.parameters():
            p.requires_grad = False
        lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj","self_attn.out_proj"]
        target_modules = []
        start_layer_idx = np.max([self.num_layer-self.declayers,0])
        for idx in range(start_layer_idx,self.num_layer):
            for layer_num in lora_targets:
                target_modules.append(f"layers.{idx}.{layer_num}")
        lora_config = LoraConfig(
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none"
        )
        self.lora_esm = get_peft_model(self.esm2, lora_config)
        self.liners = nn.Sequential(
            nn.Linear(1280, 256), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.esm_dim))
    def forward(self, X, seq_len):
        outputs = self.lora_esm(X, repr_layers=[self.num_layer], return_contacts=False)
        residue_feat = outputs["representations"][self.num_layer]
        residue_feat = residue_feat.squeeze(0)[1:seq_len+1,:]
        esm_feat = self.liners(residue_feat)
        return esm_feat
    
class VectorQuantizer(nn.Module):
	"""
	VQ-VAE layer: Input any tensor to be quantized.
	Args:
		embedding_dim (int): the dimensionality of the tensors in the
		quantized space. Inputs to the modules must be in this format as well.
		num_embeddings (int): the number of vectors in the quantized space.
		commitment_cost (float): scalar which controls the weighting of the loss terms.
	"""

	def __init__(self, args):
		super(VectorQuantizer,self).__init__()
		self.embedding_dim = args['embedding_dim']
		self.num_embeddings = args['num_embeddings']
		self.commitment_cost = args['commitment_cost']

		# initialize embeddings
		self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

	def forward(self, x):
		x = F.normalize(x, p=2, dim=-1)
		encoding_indices = self.get_code_indices(x)
		quantized = self.quantize(encoding_indices)

		q_latent_loss = F.mse_loss(quantized, x.detach())
		e_latent_loss = F.mse_loss(x, quantized.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Straight Through Estimator
		quantized = x + (quantized - x).detach().contiguous()

		return quantized, loss, encoding_indices

	def get_code_indices(self, x):
		distances = (
				torch.sum(x ** 2, dim=-1, keepdim=True) +
				torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
				2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
		)

		encoding_indices = torch.argmin(distances, dim=1)

		return encoding_indices

	def quantize(self, encoding_indices):
		"""Returns embedding tensor for a batch of indices."""
		return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)
    
class vqcodebook_CL(nn.Module):
    def __init__(self, args):
        super(vqcodebook_CL, self).__init__()
        self.vq = VectorQuantizer(args)
        self.gnn_encoder = GNNEncoder(args)
        self.gnn_decoder = GNNDecoder(args)
        self.esm_encoder = esm_encoder(args)
        self.contra_loss = InfoNCELoss()
        self.criterion = nn.MSELoss()
    
    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, batch_token,seq_len,batch_graph):
        x = batch_graph.ndata['x']
        esm_feats = self.esm_encoder(batch_token, seq_len)
        h = self.gnn_encoder(batch_graph,x)
        z_q,eq_loss,indices = self.vq(h)
        # z_q_ste = z_q + (h - z_q).detach()
        #计算对比损失
        contra_loss = self.contra_loss(esm_feats, z_q)
        x_recon = self.gnn_decoder(batch_graph, z_q)
        recon_loss = self.criterion(x_recon, x)
        loss = recon_loss + eq_loss + contra_loss
        
        return loss,recon_loss,eq_loss,contra_loss
    
class vqcodebook(nn.Module):
    def __init__(self,args):
        super(vqcodebook,self).__init__()
        self.mask_rate = args['mask_rate']
        self.num_masking = args['num_masking']
        self.remask_rate = args['remask_rate']
        self.input_size = args['input_dim']

        self.embedding_dim = args['embedding_dim']
        self.num_embeddings = args['num_embeddings']
        self.commitment_cost = args['commitment_cost']

        self.vq = VectorQuantizer(args)
        self.gnn_encoder = GNNEncoder(args)
        self.gnn_decoder = GNNDecoder(args)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_size))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.embedding_dim))
        self.reset_parameters_for_token()
        self.criterion_mse = self.setup_loss_fn(loss_fn='mse', alpha_l=0.0)
        self.criterion_sce = self.setup_loss_fn(loss_fn='sce', alpha_l=args['alpha_l'])

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    def encoding_mask_noise(self, g, x, mask_rate):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        token_node = mask_nodes
        out_x[mask_nodes] = 0.0
        out_x[token_node] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, mask_nodes, keep_nodes
    def forward(self, batch_graph):
        x = batch_graph.ndata['x']
        #不使用掩码
        z = self.gnn_encoder(batch_graph,x)
        e,e_q_loss,encoding_indices = self.vq(z)
        x_recon = self.gnn_decoder(batch_graph,e)
        recon_loss = self.criterion_mse(x_recon,x)
        #使用掩码

        # use_g, out_x, mask_nodes, keep_nodes = self.encoding_mask_noise(batch_graph, x, self.mask_rate)
        # z = self.gnn_encoder(batch_graph,out_x)
        # e,e_q_loss,encoding_indices = self.vq(z)
        # x_recon = self.gnn_decoder(batch_graph,e)
        # recon_loss = self.criterion_sce(x_recon[mask_nodes],x[mask_nodes])
        # recon_loss = self.criterion_mse(x_recon,x)

        loss_rec_all = 0
        for i in range(self.num_masking):
            mask = torch.bernoulli(
				torch.full(size=(self.num_embeddings,), fill_value=self.remask_rate)).bool()
            mask_index = mask[encoding_indices]
            e[mask_index] = 0.0
            e[mask_index] += self.dec_mask_token
            x_remask_recon = self.gnn_decoder(batch_graph, e)
            #不使用
            x_init = x_remask_recon[mask_index]
            x_rec = batch_graph.ndata['x'][mask_index]
            #使用
            # x_init = x_remask_recon[mask_nodes]
            # x_rec = batch_graph.ndata['x'][mask_nodes]
            loss_recn = self.criterion_sce(x_init,x_rec)
            loss_rec_all += loss_recn
        loss_rec_remask = loss_rec_all / int(self.num_masking)
        loss = loss_rec_remask+e_q_loss+recon_loss
        return loss,loss_rec_remask,e_q_loss,recon_loss





