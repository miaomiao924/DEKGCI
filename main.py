#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from tqdm import tqdm #产生进度条
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score
import matplotlib.pyplot as plt


import torch.optim as optim
from NGCF.utility import helper
from NGCF.utility.helper import *
from NGCF.utility.batch_test import *
from NGCF import NG_KGCN
from NGCF.NG_KGCN import NG_KGCN

import warnings
import dataloader4kg
from NGCF.utility.batch_test import data_generator
from KGCN import KGCN
from NGCF import NGCF

warnings.filterwarnings('ignore')
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
t0 = time()


# def do_evaluate(model, testSet):
#     testSet = torch.LongTensor(testSet)
#     model.eval()
#     with torch.no_grad():
#         user_ids = testSet[:, 0]
#         item_ids = testSet[:, 1]
#         labels = testSet[:, 2]
#         logits = model(data_generator.n_users,
#                 data_generator.n_items,user_ids, item_ids, True)
#         predictions = [1 if i >= 0.5 else 0 for i in logits]
#         p = precision_score(y_true=labels, y_pred=predictions)
#         r = recall_score(y_true=labels, y_pred=predictions)
#         acc = accuracy_score(labels, y_pred=predictions)
#         return p, r, acc



def train( epochs, batchSize, lr, n_user, n_item, norm_adj, n_users, n_entitys, n_relations,
      adj_entity, adj_relation,
      train_set, test_set,
      n_neighbors,
      aggregator_method='sum',
      act_method=F.relu, drop_rate=0, weight_decay=5e-4
      ):
    #print('norm_adj',norm_adj)
    model = NG_KGCN(data_generator.n_users,
                    data_generator.n_items,
                    norm_adj,args,n_users, n_entitys, n_relations,
                     adj_entity, adj_relation,
                    n_neighbors=n_neighbors, e_dim=16,
                    aggregator_method=aggregator_method,
                    act_method=act_method,
                    drop_rate=drop_rate).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = nn.BCELoss()
    dataIter = dataloader4kg.DataIter()
    print(len(train_set) // batchSize)

    for epoch in range(epochs):
        total_loss = 0.0
        for datas in tqdm(dataIter.iter(train_set, batchSize=batchSize)):
            #print('datas',datas.shape)
            #print("hello")
            user_ids = datas[:, 0]
            item_ids = datas[:, 1]
           # print('uid', user_ids)
           # print('iid', item_ids)
            labels = torch.tensor(datas[:, 2]).cuda()
            logits = model.forward(data_generator.n_users,
                                   data_generator.n_items, user_ids, item_ids)

           # print(logits, labels.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
       #print(test_set)
        p, r, acc, f1 ,auc = model.do_evaluate(model,test_set)

        print("Epoch {} | Loss {:.4f} | Precision {:.4f} | Recall {:.4f} | Accuracy {:.4f}  | F1 {:.4f} | AUC {:.4f}"
              .format(epoch, total_loss / (len(train_set) // batchSize), p, r, acc, f1,auc))
       #  print("Epoch {} | Loss {:.4f} | Precision {:.4f} | Recall {:.4f} | Accuracy {:.4f}"
       #        .format(epoch, total_loss / (len(train_set) // batchSize), p, r, acc))

        loss_f.append(total_loss / (len(train_set) // batchSize))
        pre.append(p)
        recall.append(r)
        acc1.append(acc)
        AUC.append(auc)
        F1.append(f1)

    print('best_loss', min(loss_f))
    print('best_pre', max(pre))
    print('best_recall', max(recall))
    print('best_acc', max(acc1))
    print('best_f1', max(F1))
    print('best_auc', max(AUC))



if __name__ == '__main__':
    print(torch.cuda.is_available())
    args.device:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 用GPU跑数据
    print(device.type)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()  # 得到邻接矩阵 在load_data第83行

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    # 这是KGCN的初始化
    n_neighbors = 8


    users, items, train_set, test_set = dataloader4kg.readRecData(dataloader4kg.Ml_100K.RATING)
    entitys, relations, kgTriples = dataloader4kg.readKgData(dataloader4kg.Ml_100K.KG)
    #print('entity',entitys)
    adj_kg = dataloader4kg.construct_kg(kgTriples)
    adj_entity, adj_relation = dataloader4kg.construct_adj(n_neighbors, adj_kg, len(entitys))

    loss_f = []
    pre = []
    recall = []
    acc1 = []
    AUC = []
    F1 = []


    train(epochs=100, batchSize=32, lr=5e-4,n_user=data_generator.n_users,
          n_item=data_generator.n_items, norm_adj=norm_adj,
          n_users=max(users) + 1, n_entitys=max(entitys) + 1,
          n_relations=max(relations) + 1, adj_entity=adj_entity,
          adj_relation=adj_relation, train_set=train_set,
          test_set=test_set, n_neighbors=n_neighbors,
          aggregator_method='sum', act_method=F.relu, drop_rate=0, weight_decay=1e-4)

    # 验证
    x = range(100)
    ax = plt.gca()
    plt.plot(x, loss_f, 'b', label="loss")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("loss")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, pre, 'b', label="pre")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("pre")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, recall, 'b', label="recall")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("recall")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, acc1, 'b', label="acc")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("acc")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, F1, 'b', label="F1")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("F1")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, AUC, 'b', label="auc")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("auc")
    plt.legend(loc='upper right')
    plt.show()
















