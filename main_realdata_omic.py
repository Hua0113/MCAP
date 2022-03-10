import pandas as pd
import numpy as np
import csv

from mcap import train_CoxKmeans
from mcap import predictCoxKmeans
from mcap import getCindex

import torch
from support import get_omic_data
from support import sort_data
from support import mkdir
from support import plot_curve
from support import cal_pval
from idle_gpu import idle_gpu

#gpu_id = idle_gpu()
#device = torch.device("cuda:{}".format(idle_gpu()) if torch.cuda.is_available() else "cpu")
#print(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



is_plot = False
## is drawing
nn_config = {
    "learning_rate": 0.0000007, #0.0000007
    "learning_rate_decay": 0.999,
    "activation": 'relu',
    "epoch_num": 500,
    "skip_num": 5,
    "L1_reg": 1e-5,
    "L2_reg": 1e-5,
    "optimizer": 'Adam',
    "dropout": 0.0,
    "hidden_layers": [1000, 500, 24, 1],
    "standardize":True,
    "batchnorm":False,
    "momentum": 0.9,
    "n_clusters" : 2,
    "update_interval" : 1,
    "kl_rate":10,
    "ae_rate":1,
    "seed": 1
}



dataset = ["my_dataset/ov2"]

# ### 5-independent TEST
test_pred_set = []
test_y_set = []
mse_set = []
ss_set = []
classify_set = []
x_reconstuct_set = []

# hidden_l = [20]
# hidden_l = [100, 50, 20]
# lr_set = [1E-7,5E-7,1E-6]

hidden_l = [50]
lr_set = [1E-6]

for filename in dataset:
    for h in hidden_l:
        for lr in lr_set:
            mse_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            ss_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            classify_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            x_reconstuct_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            for seed in range(1):
                average_Cindex = 0
                for fold_num in range(5):
                    ori_train_X, ori_train_Y, ori_test_X, ori_test_Y = get_omic_data(fea_filename=(filename + ".csv"), seed=seed, nfold = 5 ,fold_num=fold_num)
                    ori_idx, ori_train_X, ori_train_Y = sort_data(ori_train_X, ori_train_Y)
                    input_nodes = len(ori_train_X[0])
                    nn_config["learning_rate"] = lr
                    nn_config["hidden_layers"][2] = h

                    outputlist, model = train_CoxKmeans(device, nn_config, input_nodes, ori_train_X, ori_train_Y, ori_test_X, ori_test_Y)

                    mse_set.append(outputlist[0])
                    ss_set.append(outputlist[1])
                    classify_set.append(outputlist[2])
                    x_reconstuct_set.append(outputlist[3])

                    test_x_bar, test_q, prediction = predictCoxKmeans(model, device, nn_config, ori_test_X)
                    test_pred_set.append(prediction)
                    test_y_set.append(ori_test_Y)
                    Cindex = getCindex(ori_test_Y,prediction)
                    print("Cindex is : " + str(Cindex))
                    average_Cindex += Cindex
                print("Average Cindex is : "+str(average_Cindex/5))
            mkdir("result")
            with open("result/" + "prediction" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(test_pred_set)
            with open("result/" + "test_Y" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(test_y_set)
            with open("result/" + "testmse" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(mse_set)
            with open("result/" + "SS" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(ss_set)
            with open("result/" + "classify" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(classify_set)
            with open("result/" + "reconsturct" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(x_reconstuct_set)

