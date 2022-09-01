import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm #产生进度条
import dataloader4kg
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score

class KGCN( nn.Module ):

    def __init__( self, n_users, n_entitys, n_relations,
                  adj_entity, adj_relation, n_neighbors,  e_dim= 16,
                  aggregator_method = 'sum',
                  act_method = F.relu, drop_rate=0.5):
        super(KGCN, self).__init__()
        self.e_dim = e_dim  # 特征向量维度
        self.aggregator_method = aggregator_method #消息聚合方法
        self.n_neighbors = n_neighbors #邻居的数量
        self.user_embedding = nn.Embedding( n_users, e_dim, max_norm = 1 )
        self.entity_embedding = nn.Embedding( n_entitys, e_dim, max_norm = 1)
        self.relation_embedding = nn.Embedding( n_relations, e_dim, max_norm = 1)

        self.adj_entity = adj_entity #节点的邻接列表
        self.adj_relation = adj_relation #关系的邻接列表

        #线性层
        self.linear_layer = nn.Linear(
                in_features = self.e_dim * 2 if self.aggregator_method == 'concat' else self.e_dim,
                out_features = self.e_dim,
                bias = True)

        self.act = act_method #激活函数
        self.drop_rate = drop_rate #drop out 的比率

    def forward(self, users, items, is_evaluate = False):
        neighbor_entitys, neighbor_relations = self.get_neighbors( items )
        user_embeddings = self.user_embedding(users.cuda())
        item_embeddings = self.entity_embedding(items.cuda())
        #print('item',items.shape)
        #print('item_embedding', item_embeddings.shape)
        #得到v波浪线
        neighbor_vectors = self.__get_neighbor_vectors( neighbor_entitys, neighbor_relations, user_embeddings )
        #进行自身消息聚合
        out_item_embeddings = self.aggregator( item_embeddings, neighbor_vectors,is_evaluate)
        #激活函数进行点击预测
        #print('out',out_item_embeddings.shape)
        #out = torch.sigmoid( torch.sum( user_embeddings * out_item_embeddings, axis = -1 ) )
        #输出
        return user_embeddings,out_item_embeddings

    def get_neighbors( self, items ):#得到邻居的节点embedding,和关系embedding
        #[[1,2,3,4,5],[2,1,3,4,5]...[]]#总共batchsize个n_neigbor的id
        #print('item',items)
        entity_ids = [ self.adj_entity[item] for item in items ]
        relation_ids = [ self.adj_relation[item] for item in items ]
        neighbor_entities = [ torch.unsqueeze(self.entity_embedding(torch.LongTensor(one_ids).cuda()),0) for one_ids in entity_ids]
        neighbor_relations = [ torch.unsqueeze(self.relation_embedding(torch.LongTensor(one_ids).cuda()),0) for one_ids in relation_ids]
        # [batch_size, n_neighbor, dim]
        neighbor_entities = torch.cat( neighbor_entities, dim=0 )
        neighbor_relations = torch.cat( neighbor_relations, dim=0 )

        return neighbor_entities, neighbor_relations

    #得到v波浪线
    def __get_neighbor_vectors(self, neighbor_entitys, neighbor_relations, user_embeddings):
        # [batch_size, n_neighbor, dim]
        user_embeddings = torch.cat([torch.unsqueeze(user_embeddings,1) for _ in range(self.n_neighbors)],dim=1)
        # [batch_size, n_neighbor]
        user_relation_scores = torch.sum(user_embeddings * neighbor_relations, axis=2)
        # [batch_size, n_neighbor]
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)
        # [batch_size, n_neighbor, 1]
        user_relation_scores_normalized = torch.unsqueeze(user_relation_scores_normalized, 2)
        # [batch_size, dim]
        neighbor_vectors = torch.sum(user_relation_scores_normalized * neighbor_entitys, axis=1)
        return neighbor_vectors

    #经过进一步的聚合与线性层得到v
    def aggregator(self,item_embeddings, neighbor_vectors, is_evaluate):
        # [batch_size, dim]
        if self.aggregator_method == 'sum':
            output = item_embeddings + neighbor_vectors
            #print('agg', item_embeddings.shape)
            #print('agg', neighbor_vectors.shape)
        elif self.aggregator_method == 'concat':
            # [batch_size, dim * 2]
            output = torch.cat([item_embeddings, neighbor_vectors], axis=-1)
            #print('agg', output.shape)
        else:#neighbor
            output = neighbor_vectors
       # print('agg',output.shape)
        if not is_evaluate:
            output = F.dropout(output, self.drop_rate)
        # [batch_size, dim]
        output = self.linear_layer(output)
        return self.act(output)  #输出item-embedding




