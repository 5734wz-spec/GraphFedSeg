import logging
import numpy as np
import torch.nn as nn
import pdb
import torch
import torch.nn.functional as F
# from utils.loss_func import EDL_Dice_Loss
# from utils.loss_func import Train_Dice_Loss
from utils.loss_func import GraphAwareLoss
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# def train_fedevi(round_idx, client_idx, model, dataloader, optimizer, args):
#     model.train()

#     max_epoch = args.max_epoch

#     seg_loss = EDL_Dice_Loss(kl_weight=args.kl_weight, annealing_step=args.annealing_step)


#     for epoch in range(max_epoch):
#         for iters, (_, data) in enumerate(dataloader):
#             optimizer.zero_grad()

#             image, label = data['image'], data['label']
#             image = image.cuda()
#             label = label.cuda()

#             logit = model(image)[0]
#             loss = seg_loss(logit, label, round_idx)

#             loss.backward()
#             optimizer.step()

#     return model
            

# def train_graph(round_idx, client_idx, model, dataloader, optimizer, args, cluster_model=None):
#     model.train()
#     # 使用与 train_fedevi 相同的损失函数
#     seg_loss = EDL_Dice_Loss(kl_weight=args.kl_weight, annealing_step=args.annealing_step)
    
#     for epoch in range(args.max_epoch):
#         for iters, (_, data) in enumerate(dataloader):  # 调整数据加载方式与 train_fedevi 一致
#             optimizer.zero_grad()

#             image, label = data['image'], data['label']
#             image = image.cuda()
#             label = label.cuda()

#             logit = model(image)[0]  # 假设模型输出结构与 train_fedevi 一致
#             loss = seg_loss(logit, label, round_idx)

#             # 添加pFedGraph的余弦相似度约束
#             if round_idx > 0 and cluster_model is not None:
#                 flatten_model = torch.cat([p.reshape(-1) for p in model.parameters()])
#                 flatten_cluster = torch.cat([p.reshape(-1) for p in cluster_model.parameters()])
#                 loss += args.lam * (1 - torch.nn.functional.cosine_similarity(
#                     flatten_cluster.unsqueeze(0),
#                     flatten_model.unsqueeze(0)
#                 ))
            
#             loss.backward()
#             optimizer.step()
    
#     return model

# def train_graph(round_idx, client_idx, model, dataloader, optimizer, args):
#     model.train()
#     total_uncertainty = 0.0
#     seg_loss = EDL_Dice_Loss(kl_weight=args.kl_weight, annealing_step=args.annealing_step)
    
#     for epoch in range(args.max_epoch):
#         for iters, (_, data) in enumerate(dataloader):
#             optimizer.zero_grad()
            
#             # 数据加载
#             image = data['image'].cuda()
#             label = data['label'].cuda()
            
#             # 前向传播
#             logit = model(image)[0]
            
#             # ========== 新增不确定性计算 ==========
#             # 当前熵计算方式
#             prob = torch.softmax(logit, dim=1)
#             entropy = -torch.sum(prob * torch.log(prob + 1e-6), dim=1)
#             batch_uncertainty = entropy.mean()
#             total_uncertainty += batch_uncertainty.item()

            
#             # 损失计算
#             loss = seg_loss(logit, label, round_idx)
#             loss.backward()
#             optimizer.step()
    
#     # 计算平均不确定性
#     avg_uncertainty = total_uncertainty / (len(dataloader) * args.max_epoch)
#     return model, avg_uncertainty



# def train_graph(round_idx, client_idx, model, dataloader, optimizer, args):
#     model.train()
#     max_epoch = args.max_epoch
#     total_uncertainty = 0.0
#     seg_loss = Train_Dice_Loss()
    
#     for epoch in range(args.max_epoch):
#         for iters, (_, data) in enumerate(dataloader):
#             optimizer.zero_grad()
            
#             # 数据加载
#             image = data['image'].cuda()
#             label = data['label'].cuda()
            
#             # 前向传播
#             logit = model(image)[0]
            
#             # ========== 混合不确定性计算 ==========
#             with torch.no_grad():  # 新增：不计算梯度
#                 evidence = torch.exp(logit)
#                 alpha = evidence + 1
#                 S = torch.sum(alpha, dim=1, keepdim=True)
#                 uncertainty = (alpha.shape[1] / S.squeeze(1)).mean()  # 添加均值计算
            
     
#             # 损失计算（仅使用原始损失）
#             # loss = seg_loss(logit, label, round_idx)
#             loss = seg_loss(logit, label)
#             loss.backward()
#             optimizer.step()
            
#             total_uncertainty += uncertainty.item()  # 累积不确定性
    
#     avg_uncertainty = total_uncertainty / (len(dataloader) * args.max_epoch)
#     return model, avg_uncertainty


# def train_graph(round_idx, client_idx, model, dataloader, optimizer, args, global_model=None):
#     model.train()
#     max_epoch = args.max_epoch
#     total_uncertainty = 0.0
    
#     # 初始化复合损失函数
#     seg_loss = GraphAwareLoss(lambda_graph=args.lam, lambda_uncertainty=args.uncertainty_weight)
    
#     for epoch in range(args.max_epoch):
#         for iters, (_, data) in enumerate(dataloader):
#             optimizer.zero_grad()
            
#             # 数据加载
#             image = data['image'].cuda()
#             label = data['label'].cuda()
            
#             # 前向传播
#             logit = model(image)[0]
            
#             # ===== 混合不确定性计算 =====
#             with torch.no_grad():
#                 evidence = torch.exp(logit)
#                 alpha = evidence + 1
#                 S = torch.sum(alpha, dim=1, keepdim=True)
#                 batch_uncertainty = (alpha.shape[1] / S.squeeze(1)).mean()

#             # ===== 统一使用复合损失 =====
#             total_loss = seg_loss(
#                 pred=logit,
#                 label=label,
#                 current_model=model,
#                 global_model=global_model,
#                 uncertainty=batch_uncertainty
#             )
            
#             # 反向传播
#             total_loss.backward()
#             optimizer.step()
            
#             total_uncertainty += batch_uncertainty.item()
    
#     avg_uncertainty = total_uncertainty / (len(dataloader) * args.max_epoch)
#     return model, avg_uncertainty

def train_graph(round_idx, client_idx, model, dataloader, optimizer, args, neighbor_indices, graph_matrix, global_model=None, cluster_vectors=None,local_models=None):
    model.train()
    max_epoch = args.max_epoch
    total_uncertainty = 0.0
    
    # 初始化复合损失函数
    seg_loss = GraphAwareLoss(lambda_graph=args.lam, lambda_uncertainty=args.uncertainty_weight)
    
    for epoch in range(args.max_epoch):
        for iters, (_, data) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 数据加载
            image = data['image'].cuda()
            label = data['label'].cuda()
            
            # 前向传播
            logit = model(image)[0]
            
            # ===== 混合不确定性计算 =====
            with torch.no_grad():
                evidence = torch.exp(logit)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                batch_uncertainty = (alpha.shape[1] / S.squeeze(1)).mean()
                

            # ===== 新增个性化正则项 =====
            if cluster_vectors is not None:
                # 将当前模型参数扁平化
                current_params = torch.cat([p.view(-1) for p in model.parameters()])
                # 获取对应的聚类中心向量
                cluster_center = cluster_vectors[client_idx].cuda()
                # 计算余弦相似度正则项
                similarity_loss = 1 - F.cosine_similarity(current_params.unsqueeze(0), 
                                                         cluster_center.unsqueeze(0))
                similarity_loss = args.graph_fusion_ratio * similarity_loss
            else:
                similarity_loss = 0

            # ===== 统一使用复合损失 =====
            total_loss = seg_loss(
                pred=logit,
                label=label,
                current_model=model,
                neighbor_models=[local_models[k] for k in neighbor_indices],  # 使用传入的local_models  # 从主程序获取邻居索引
                graph_weights=graph_matrix[client_idx][neighbor_indices],     # 协作权重
                uncertainty=batch_uncertainty
    )
            
            # 反向传播
            total_loss.backward()
            # ===== 新增梯度稳定机制 =====
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
            optimizer.step()
            # 参数约束（在优化器step之后）
            # with torch.no_grad():
            #     for param in model.parameters():
            #         param.data = torch.clamp(param, -5, 5)
            
            total_uncertainty += batch_uncertainty.item()
    
    avg_uncertainty = total_uncertainty / (len(dataloader) * args.max_epoch)
    return model, avg_uncertainty