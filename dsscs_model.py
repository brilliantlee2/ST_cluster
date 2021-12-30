#Dual self-supervised cluster for spatial data.
#author:Liyiyan
#date:20211207

##不进行预训练

from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
from utils_func import adata_preprocess
from args_parser import set_parser


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)        #2000->500
        self.enc_2 = Linear(n_enc_1, n_enc_2)        #500->500
        self.enc_3 = Linear(n_enc_2, n_enc_3)        #500->2000
        self.z_layer = Linear(n_enc_3, n_z)          #2000->10

        self.dec_1 = Linear(n_z, n_dec_1)            #10->2000
        self.dec_2 = Linear(n_dec_1, n_dec_2)        #2000->500
        self.dec_3 = Linear(n_dec_2, n_dec_3)        #500->500
        self.x_bar_layer = Linear(n_dec_3, n_input)  #500->2000

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z  #x_bar是重构的表达矩阵  z是隐藏空间纬度


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):  #n_input是表达矩阵，AE用来重构表达矩阵？
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        #self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)   #2000->500
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)   #500->500
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)   #500->2000
        self.gnn_4 = GNNLayer(n_enc_3, n_z)       #2000->10
        self.gnn_5 = GNNLayer(n_z, n_clusters)    #10->4

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()




### 

def train_sdcn(dataset):
    model = SDCN(300, 300, 500, 500, 300, 300,
                n_input=args.n_input,               #default==500   
                n_z=args.n_z,                       #default==30
                n_clusters=args.n_clusters,         #default==8
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)



    # KNN Graph
    adata = load_data(151673)
    adj = load_graph(adata)   #adj是一个稀疏格式的矩阵，to_dense()查看原样
    adj = adj.cuda()

    # cluster parameter initiate
    adata_X = adata_preprocess(adata, min_cells=5, pca_n_comps=params.cell_feat_dim)
    adata_X = torch.Tensor(adata_X).to(device)
    y = convert_str_to_int(adata)                               ###把字符串类变成数子类，20212114
    with torch.no_grad():
        _, _, _, _, z = model.ae(adata.X)   #z是十维

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')


    for epoch in range(200):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(adata.X, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    
    args = set_parser()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    
    dataset = load_data(151673)


    print(args)
    train_sdcn(dataset)



