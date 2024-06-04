import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm #产生进度条
import dataloader4kg
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score
from NGCF.utility.aggregator import Aggregator
class KGCN( nn.Module ):

    def __init__( self, n_users, n_entitys, n_relations,
                  adj_entity, adj_relation, n_neighbors,  e_dim= 64,
                  aggregator_method = 'sum',
                  act_method = F.relu, drop_rate=0.5):
        super(KGCN, self).__init__()
        self.e_dim = e_dim  # 特征向量维度
        self.aggregator_method = aggregator_method #消息聚合方法
        self.n_neighbors = n_neighbors #邻居的数量
        self.user_embedding = nn.Embedding( n_users, e_dim, max_norm = 1 )
        self.entity_embedding = nn.Embedding( n_entitys, e_dim, max_norm = 1)
        self.relation_embedding = nn.Embedding( n_relations, e_dim, max_norm = 1)
        self.n_iter=1
        self.batch_size=8
        self.adj_entity = adj_entity #节点的邻接列表
        self.adj_relation = adj_relation #关系的邻接列表
        self.aggregator = Aggregator(self.batch_size, self.e_dim, self.aggregator_method)
        #线性层
        self.linear_layer = nn.Linear(
                in_features = self.e_dim * 2 if self.aggregator_method == 'concat' else self.e_dim,
                out_features = self.e_dim,
                bias = True)

        self.act = act_method #激活函数
        self.drop_rate = drop_rate #drop out 的比率

    def forward(self, users, items, is_evaluate = False):

        batch_size = items.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        users = users.view((-1, 1))
        items = items.view((-1, 1))

        user_embeddings = self.user_embedding(users.cuda())
        item_embeddings = self.entity_embedding(items.cuda())
        neighbor_entitys, neighbor_relations = self.get_neighbors(items)  # 得到不同距离的实体集和关系集合
        out_item_embeddings = self._aggregate(user_embeddings, neighbor_entitys, neighbor_relations)
        return user_embeddings,out_item_embeddings

    def get_neighbors(self, items):
        entities = [items]
        relations = []

        for h in range(self.n_iter):
            #neighbor_entities = torch.LongTensor(self.adj_entity[entities[h]]).view((self.batch_size, -1))
            neighbor_entities = torch.LongTensor(self.adj_entity[entities[h]])
            #print('1',neighbor_entities.shape)
            neighbor_entities=neighbor_entities.view(self.batch_size,-1)
            #print('2', neighbor_entities.shape)
            neighbor_relations = torch.LongTensor(self.adj_relation[entities[h]]).view((self.batch_size, -1))
            #print('3', neighbor_relations.shape)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        #print(entities)

        return entities, relations
   

    def _aggregate(self, user_embeddings, entities, relations):

        entity_vectors = [ torch.unsqueeze(self.entity_embedding(torch.LongTensor(one_ids).cuda()),0) for one_ids in entities]
        relation_vectors = [ torch.unsqueeze(self.relation_embedding(torch.LongTensor(one_ids).cuda()),0) for one_ids in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    entity_vectors[hop],  #自己的向量表示
                    entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbors, self.e_dim)), #邻居实体向量表示
                    relation_vectors[hop].view((self.batch_size, -1, self.n_neighbors, self.e_dim)), #邻居关系向量表示
                    user_embeddings,  #user-embeddings
                    act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.e_dim))







