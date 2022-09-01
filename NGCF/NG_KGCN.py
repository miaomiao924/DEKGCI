
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score
from torch import device

import dataloader4kg
from NGCF.utility.batch_test import data_generator, args
from KGCN import KGCN
from NGCF import NGCF
class NG_KGCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj,args,n_users, n_entitys, n_relations,
                  adj_entity, adj_relation, n_neighbors, e_dim = 16,
                  aggregator_method = 'sum',
                  act_method = F.relu, drop_rate=0.5):
        super(NG_KGCN, self).__init__()  #NGCF
        self.args= args
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.device = device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        #self.init_weight = NGCF.init_weight(self)
        self.e_dim = e_dim  # 特征向量维度
        self.aggregator_method = aggregator_method  # 消息聚合方法
        self.n_neighbors = n_neighbors  # 邻居的数量
        self.user_embedding = nn.Embedding(n_users, e_dim, max_norm=1)
        self.entity_embedding = nn.Embedding(n_entitys, e_dim, max_norm=1)
        self.relation_embedding = nn.Embedding(n_relations, e_dim, max_norm=1)

        self.adj_entity = adj_entity  # 节点的邻接列表
        self.adj_relation = adj_relation  # 关系的邻接列表
        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = NGCF.NGCF.init_weight(self)

        """
        *********************************************************
        Get sparse adj.
        """
        #self.sparse_norm_adj = NGCF.NGCF._convert_sp_mat_to_sp_tensor(self,self.norm_adj).to(self.device)
        self.sparse_norm_adj = NGCF.NGCF._convert_sp_mat_to_sp_tensor(self, self.norm_adj).to(args.device)
        #这里不确定加不加 .to(device)
        self.model1 = NGCF.NGCF(data_generator.n_users,
                      data_generator.n_items,
                      self.norm_adj,
                      args).to(args.device)

        self.model2 = KGCN(n_users, n_entitys, n_relations,
                      adj_entity, adj_relation,
                      n_neighbors=n_neighbors, e_dim= 16,
                      aggregator_method=aggregator_method,
                      act_method=act_method,
                      drop_rate=drop_rate,
                          ).to(args.device)

        #耦合 mlp



    def forward(self, n_user, n_item, n_users,n_entitys,is_evaluate = False):

        u1_g_embeddings = self.model1(n_users, drop_flag=args.node_dropout_flag)
        u2_g_embeddings, i2_g_embeddings = self.model2(n_users, n_entitys,is_evaluate = False) #KGCN 传过去的是id


        out = torch.sigmoid(torch.sum(torch.multiply(u1_g_embeddings, i2_g_embeddings), axis=-1))

        return out

    def do_evaluate(self, model: object, testSet: object) -> object:
        testSet = torch.LongTensor(testSet)
        model.eval()
        with torch.no_grad():
            user_ids = testSet[:, 0]
            item_ids = testSet[:, 1]
            labels = testSet[:, 2]
            logits = model(data_generator.n_users,
                    data_generator.n_items,user_ids, item_ids, True)
            predictions = [1 if i >= 0.5 else 0 for i in logits]
            p = precision_score(y_true=labels, y_pred=predictions)
            r = recall_score(y_true=labels, y_pred=predictions)
            acc = accuracy_score(labels, y_pred=predictions)
            auc = roc_auc_score(y_true=labels, y_score=logits.cpu().numpy())
            f1 = f1_score(y_true=labels, y_pred=predictions)
            return p, r, acc,  f1, auc


