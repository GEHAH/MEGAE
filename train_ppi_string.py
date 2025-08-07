import time,yaml, json,copy
import math, random,sys
import argparse
import os, csv
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('/data2/csz/PPI_tnnls/data')
from data.dataset_string import Pretrain_VQ_Dataset,load_dataset,collate


sys.path.append('/data2/csz/PPI_tnnls/Models')
from Models.model_utils import evaluat_metrics,check_writable,set_seed
from Models.model_vq import vqcodebook
from Models.model_ppi import PPI_model

import warnings
warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluator(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, mode='metric'):
	eval_output_list = []
	eval_labels_list = []

	batch_num = math.ceil(len(index) / batch_size)

	model.eval()

	with torch.no_grad():
		for batch in range(batch_num):
			if batch == batch_num - 1:
				eval_idx = index[batch * batch_size:]
			else:
				eval_idx = index[batch * batch_size:(batch + 1) * batch_size]

			output = model(ppi_g, prot_embed,ppi_list, eval_idx)
			eval_output_list.append(output.detach().cpu())
			eval_labels_list.append(labels[eval_idx].detach().cpu())

		f1_score, aupr = evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))

	if mode == 'metric':
		return f1_score, aupr
	elif mode == 'output':
		return torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0)

def train_one_epoch(model, ppi_g, prot_embed,ppi_list, labels, index, batch_size, optimizer, loss_fn):
	f1_sum = 0.0
	loss_sum = 0.0

	batch_num = math.ceil(len(index) / batch_size)
	random.shuffle(index)

	model.train()
	for batch in range(batch_num):
		if batch == batch_num - 1:
			train_idx = index[batch * batch_size:]
		else:
			train_idx = index[batch * batch_size:(batch + 1) * batch_size]
		output = model(ppi_g, prot_embed,ppi_list, train_idx)
		loss = loss_fn(output, labels[train_idx])
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loss_sum += loss.item()
		f1_score, _ = evaluat_metrics(output.detach().cpu(), labels[train_idx].detach().cpu())
		f1_sum += f1_score
	# aupr_sum += aupr
	return loss_sum / batch_num, f1_sum / batch_num

def main(param,seed):
	protein_data =Pretrain_VQ_Dataset(param)
	ppi_g, ppi_list, ppi_label_list, ppi_split_dict = load_dataset(param['data_name'], param['split_mode'], seed)
	# ppi_g.to(device)
	# ppi_label_list.to(device)
	vae_model = vqcodebook(param).to(device)
	vae_model.load_state_dict(torch.load('/data2/csz/PPI_tnnls/results_vq_new/CATH43/20250625-122420/best_vq_model.pth'))
	# vae_model.load_state_dict(torch.load(r'results_PPI\CATH43_aaf7\2024-09-05 15-57-11-659\NEWVAE\newvae_model.ckpt'))
	embs = []
	data_train = DataLoader(protein_data, batch_size=1, shuffle=False,collate_fn=collate)
	vqvae=vae_model.vq
	gnn_encoder = vae_model.gnn_encoder
	vqvae.eval()
	gnn_encoder.eval()
	embs,indexs,esmfeat = [],[],[]
	for batch_graph in data_train:
		batch_graph = batch_graph.to(device)
		with torch.no_grad():
			embed_z,embed_mean,index = gnn_encoder.encoder(batch_graph,vqvae)
			embs.append(embed_mean)
		# esmfeats.append(esmf)
	prot_embed = torch.cat(embs, dim=0).to(device)
	# esm_feat = torch.cat(esmfeats, dim=0).to(device)
	del vae_model, gnn_encoder, vqvae

	torch.cuda.empty_cache()
	# esm_feat = torch.from_numpy(
	# 	np.load(f'/data/STRING_data/{param['dataset']}/ESM_feats/{param['dataset']}.npy', allow_pickle=True)).to(device)
	output_dir = "results_14/{}/{}/SEES_{}/".format(param['data_name'], timestamp, seed)
	check_writable(output_dir, overwrite=False)
	log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')
	model = PPI_model(param)
	# model = torch.nn.DataParallel(model)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=float(param['learning_rate']),
								weight_decay=float(param['weight_decay']))
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
	loss_fn = nn.BCEWithLogitsLoss().to(device)

	es = 0
	val_best = 0
	test_val = 0
	test_val_aupr = 0
	test_best = 0
	best_epoch = 0
	patience = 200  # 早停轮数

	for epoch in range(1, param["epoch"] + 1):
		train_loss, train_f1 = train_one_epoch(model, ppi_g, prot_embed, ppi_list,
											ppi_label_list, ppi_split_dict['train_index'],
											param['batch_size'], optimizer, loss_fn)
		scheduler.step(train_loss)

		if epoch % 1 == 0:
			val_f1_score, val_aupr = evaluator(model, ppi_g, prot_embed, ppi_list, ppi_label_list,
											ppi_split_dict['val_index'],
											param['batch_size'])
			test_f1_score, test_aupr = evaluator(model, ppi_g, prot_embed,ppi_list, ppi_label_list,
												ppi_split_dict['test_index'],
												param['batch_size'])
			if test_f1_score > test_best:
				test_best = test_f1_score

			if val_f1_score >= val_best:
				val_best = val_f1_score
				test_val = test_f1_score
				test_val_aupr = test_aupr
				state = copy.deepcopy(model.state_dict())
				torch.save(state, os.path.join(output_dir, "model_state.pth"))
				print(f"Best model saved with validation loss: {test_val:.4f}")
				es = 0
				best_epoch = epoch
			else:
				es += 1

			print(
				"\033[0;30;46m Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\033[0m".format(
					epoch, train_loss, train_f1, val_f1_score, test_f1_score, val_best, test_val, test_best,
					best_epoch))
			log_file.write(
				" Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\n".format(
					epoch, train_loss, train_f1, val_f1_score, test_f1_score, val_best, test_val, test_best,
					best_epoch))
			log_file.flush()
			if es >= patience:
				print("Early stopping triggered")
				model.load_state_dict(state)
				eval_output, eval_labels = evaluator(model, ppi_g, prot_embed,ppi_list, ppi_label_list,
										ppi_split_dict['test_index'],
										param['batch_size'],mode='output')

				np.save(os.path.join(output_dir, "eval_output.npy"), eval_output.detach().cpu().numpy())
				np.save(os.path.join(output_dir, "eval_labels.npy"), eval_labels.detach().cpu().numpy())
				outFile = open('PerformMetrics_Metrics_14.csv', 'a+', newline='')
				writer = csv.writer(outFile, dialect='excel')
				results = [timestamp]
				for v, k in param.items():
					results.append(k)
				results.append(str(seed))

				results.append(str(test_f1_score))
				results.append(str(test_val))
				results.append(str(test_val_aupr))
				results.append(str(test_best))
				writer.writerow(results)
	
				break
		if epoch == param["epoch"]:  # Training finishes without early stopping
			print("Training completed")
			model.load_state_dict(state)
			eval_output, eval_labels = evaluator(model, ppi_g, prot_embed, ppi_list, ppi_label_list, ppi_split_dict['test_index'], param['batch_size'], mode='output')

			np.save(os.path.join(output_dir, "eval_output.npy"), eval_output.detach().cpu().numpy())
			np.save(os.path.join(output_dir, "eval_labels.npy"), eval_labels.detach().cpu().numpy())

			outFile = open('PerformMetrics_Metrics_14.csv', 'a+', newline='')
			writer = csv.writer(outFile, dialect='excel')
			results = [timestamp]
			for v, k in param.items():
				results.append(k)
			results.append(str(seed))

			results.append(str(test_f1_score))
			results.append(str(test_val))
			results.append(str(test_val_aupr))
			results.append(str(test_best))
			writer.writerow(results)

			
	# torch.save(state, os.path.join(output_dir, "model_state.pth"))
	log_file.close()

	return test_f1_score, test_val, test_val_aupr, test_best, best_epoch

if __name__ == "__main__":
	# multiprocessing.set_start_method('spawn')
	with open('config/train_ppi.yaml') as f:
	# with open('config/Train_new.yaml') as f:
		param = yaml.load(f, Loader=yaml.FullLoader)
	for i in range(4):
		seed = 14
		set_seed(seed=seed)
		timestamp = time.strftime(
			"%Y-%m-%d %H-%M-%S") + "-%3d" % ((time.time() - int(time.time())) * 1000)
		# pre_trainvae(param,timestamp)
		test_acc, test_val, test_val_aupr, test_best, best_epoch = main(param,seed)
