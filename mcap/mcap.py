import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
import torch.nn.functional as F

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from .utils import _prepare_surv_data
from .utils import concordance_index


def _sort(X, Y):
    T = -np.abs(np.squeeze((Y)))
    sorted_idx = np.argsort(T)
    return X[sorted_idx], Y[sorted_idx]

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def standardize(x):
    for j in range(len(x[0])):
        mean = np.mean(x[:, j])
        x[:, j] = (x[:, j] - mean) / np.std(x[:, j])
    return x

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)

        # decoder
        self.dec_1 = Linear(n_enc_2, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)

        self.x_bar_layer = Linear(n_dec_2, n_input)

    def forward(self, x):

        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))

        # decoder
        dec_h1 = F.relu(self.dec_1(enc_h2))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, enc_h2


class CoxNet(nn.Module):

    def __init__(self, nn_config):
        super(CoxNet, self).__init__()

        self.layer1 = Linear(nn_config["hidden_layers"][1], nn_config["hidden_layers"][2] + 1)
        self.layer2 = Linear(nn_config["hidden_layers"][2] + 1, 1)

    def forward(self, x, k_means_label):

        # kmeans label 维度变换
        # 特征concat

        # encoder
        x1 = self.layer1(x)
        out = self.layer2(x1)
        return out

class CoxKmeans(nn.Module):
    def __init__(self,nn_config,input_nodes,pretrain_path='ae.pkl'):
        super(CoxKmeans, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path
        n_enc_1=nn_config["hidden_layers"][0]
        n_enc_2=nn_config["hidden_layers"][1]
        n_dec_1=nn_config["hidden_layers"][1]
        n_dec_2=nn_config["hidden_layers"][0]
        n_clusters = nn_config["n_clusters"]
        n_input=input_nodes
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            )

        self.layer1 = Linear(nn_config["hidden_layers"][1], nn_config["hidden_layers"][2])
        self.layer2 = Linear(nn_config["hidden_layers"][2] + 1, 1)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_enc_2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, y_pred):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        y_pred = y_pred.view(-1,1)
        embedding = self.layer1(z)
        embedding = torch.cat((embedding, y_pred),dim=1)
        risk = self.layer2(embedding)
        return x_bar, q, risk


def train_CoxKmeans(device, nn_config, input_nodes, x, y, valid_x, valid_y):
    torch.manual_seed(nn_config["seed"])
    torch.cuda.manual_seed_all(nn_config["seed"])
    np.random.seed(nn_config["seed"])
    #random.seed(nn_config["seed"])

    update_interval = nn_config["update_interval"]
    model = CoxKmeans(nn_config, input_nodes = input_nodes)
    model = model.to(device)

    if nn_config["standardize"]:
       x = standardize(x)
       valid_x = standardize(valid_x)

    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    valid_x = torch.from_numpy(valid_x.astype(np.float32))
    valid_y = torch.from_numpy(valid_y.astype(np.float32))
    x = x.to(device)
    y = y.to(device)
    valid_x = valid_x.to(device)
    valid_y = valid_y.to(device)

    optimizer = Adam(model.parameters(), lr=nn_config['learning_rate'])


    x_bar, hidden = model.ae(x)
    kmeans = KMeans(n_clusters=nn_config["n_clusters"], n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()



    outputlist = []
    test_mse_list = []
    test_SS_list = []

    y_pred = torch.from_numpy(y_pred).to(device)
    for epoch in range(nn_config["epoch_num"]):

        if epoch % update_interval == 0:

            _, tmp_q, embeeding = model(x, y_pred)
            _, hidden = model.ae(x)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            silhouetteScore = silhouette_score(hidden.data.cpu().numpy(), y_pred, metric='euclidean')


        y_pred = torch.from_numpy(y_pred).to(device)
        x_bar, q, embeeding = model(x, y_pred)
        reconstr_loss = F.mse_loss(x_bar, x)
        kl_loss = F.kl_div(q.log(), p)
        ### cox loss

        risk = embeeding

        risk_list = risk.view(-1)
        risk_exp = torch.exp(risk_list)
        t_list = y.view(-1)
        t_E = torch.gt(t_list, 0)
        risk_cumsum = torch.cumsum(risk_exp, dim=0)
        risk_cumsum_log = torch.log(risk_cumsum)
        cox_loss1 = -torch.sum(risk_list.mul(t_E))
        cox_loss2 = torch.sum(risk_cumsum_log.mul(t_E))
        cox_loss = (cox_loss1 + cox_loss2) / torch.sum(t_E)

        loss = nn_config["kl_rate"] * kl_loss + nn_config["ae_rate"] * reconstr_loss + cox_loss * 1.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % nn_config["skip_num"] == 0:
            model.eval()
            x_bar, q, train_risk = model(x, y_pred)
            test_x_bar, test_q, test_risk = predictCoxKmeans(model, device, nn_config, valid_x.detach().cpu().numpy())

            test_, test_hidden = model.ae(valid_x)
            test_reconstr_loss = F.mse_loss(test_x_bar, valid_x)
            testmse = test_reconstr_loss.data.cpu().numpy()
            silhouetteScores = silhouette_score(test_hidden.detach().cpu().numpy(),
                                                test_q.detach().cpu().numpy().argmax(1), metric='euclidean')
            test_classify = test_q.detach().cpu().numpy().argmax(1)

            test_mse_list.append(testmse)
            test_SS_list.append(silhouetteScores)

            model.train()




    outputlist.append(test_mse_list)
    outputlist.append(test_SS_list)
    outputlist.append(test_classify)
    outputlist.append(test_hidden.detach().cpu().numpy())

    return outputlist, model



def predictCoxKmeans(model, device, nn_config, input_x):
    model.eval()
    input_x = input_x.astype(np.float32)
    input_x = torch.from_numpy(input_x).to(device)
    x_bar, hidden = model.ae(input_x)
    kmeans = KMeans(n_clusters=nn_config["n_clusters"], n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    y_pred = torch.from_numpy(y_pred).to(device)
    x_bar, z, pred = model(input_x, y_pred)
    return x_bar, z, pred.detach().cpu().numpy()

def getCindex(y, prediction):
    return concordance_index(y, -prediction)





