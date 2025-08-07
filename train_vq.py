import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,random_split
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import dgl
from data.dataset import Pretrain_VQ_Dataset,collate
from Models.model_vq import vqcodebook
import numpy as np
import os,json,time,yaml
import warnings,math
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pretrain_vq(args,timestamp):
    best_train_loss = float('inf')
    patience = 20 
    patience_counter = 0

    protein_data = Pretrain_VQ_Dataset(graph_root=args['graph_root'],pdb_root=args['pdb_root'])
    # train_size = int(0.8 * len(protein_data))
    # val_size = len(protein_data) - train_size
    # train_dataset, val_dataset = random_split(protein_data, [train_size, val_size])
    train_dataloader = DataLoader(protein_data, batch_size=args['batch_size'], shuffle=True, collate_fn=collate)
    # val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate)
    output_dir = 'results_vq_new/{}/{}/'.format(args['dataname'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')
    with open(os.path.join(output_dir, "config.json"), 'a+') as tf:
        json.dump(args, tf, indent=2)
    vq_model = vqcodebook(args).to(device)
    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    for epoch in range(1,args['pre_epoch']+1):
        vq_model.train()
        train_loss = 0.
        for step,batch_graph in enumerate(train_dataloader):
            batch_graph = batch_graph.to(device)
            loss_vq, loss_rec_remask, e_q_loss, recon_loss = vq_model(batch_graph)

            print(
					"\033[0;30;43m Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f} |[0m".format(
						epoch, step, loss_vq.item(), e_q_loss.item(), recon_loss.item(), loss_rec_remask.item()))
            optimizer.zero_grad()
            loss_vq.backward()
            optimizer.step()

            train_loss += loss_vq.item()
        avg_train_loss = train_loss / len(train_dataloader)
        # scheduler.step(avg_train_loss)


        if epoch % 1 == 0:
            print("\033[0;30;43m Pre-training VQ-VAE | Epoch: {} | Train Loss: {:.5f} |\033[0m".format(epoch, avg_train_loss))
            log_file.write("Pre-training VQ-VAE | Epoch: {} | Train Loss: {:.5f} |\n".format(epoch, avg_train_loss))
            log_file.flush()

        if epoch in [30,35,40,45,50]:
            torch.save(vq_model.state_dict(), os.path.join(output_dir, f'best_vq_model{epoch}.pth'))


        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            patience_counter = 0
            torch.save(vq_model.state_dict(), os.path.join(output_dir, 'best_vq_model.pth'))
            print(f"Best model saved at epoch {epoch} with loss {best_train_loss:.5f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    log_file.close()
    print("Pre-training completed. Best validation loss: {:.5f}".format(best_train_loss))

if __name__ == "__main__":
    with open('/data2/csz/PPI_tnnls/config/codebook.yaml', 'r') as f:
        args = yaml.safe_load(f)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pretrain_vq(args, timestamp)
    print("Pre-training VQ-VAE completed.")
# -*- coding: utf-8 -*-