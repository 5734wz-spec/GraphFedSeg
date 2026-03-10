import pdb
import numpy as np
import copy
import torch

def dict_weight(dict1, weight):
    for k, v in dict1.items():
        dict1[k] = weight * v
    return dict1
    
def dict_add(dict1, dict2):
    for k, v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1




def FedAvg(global_model, local_models, client_weights, bn=False):
    global_dict = global_model.state_dict()
    
    # ========== 添加权重校验 ==========
    if abs(sum(client_weights) - 1.0) > 1e-6:
        client_weights = client_weights / client_weights.sum()
    
    # 聚合参数
    for key in global_dict.keys():
        if 'num_batches_tracked' in key and not bn:
            continue
            
        # 加权平均
        global_dict[key] = torch.sum(
            torch.stack([local_models[i].state_dict()[key].float() * client_weights[i] 
                        for i in range(len(local_models))]), 
            dim=0
        )
        
    global_model.load_state_dict(global_dict)
    return global_model

def FedUpdate(global_model, local_models, bn=True):
    if bn:
        global_dict = global_model.state_dict()
        for client_idx in range(len(local_models)):
            local_models[client_idx].load_state_dict(global_dict)
    else:
        # pdb.set_trace()
        for key in global_model.state_dict().keys():
            if 'bn' not in key:
                for client_idx in range(len(local_models)):
                    local_models[client_idx].state_dict()[key].data.copy_(global_model.state_dict()[key])

    return local_models


# 修改FedAvg函数实现层次聚合（约第300行）
def FedAvg2(global_model, local_models, client_weight, graph_matrix, bn=False):
    """
    结合图结构的层次化聚合
    """
    # 第一层：基于数据量的基础聚合
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_models[i].state_dict()[key] * client_weight[i] 
                                      for i in range(len(client_weight))], 0).sum(0)
    
    # 第二层：基于图结构的邻域增强
    # 第二层：基于图结构的邻域增强
    neighbor_dict = global_model.state_dict()
    for key in neighbor_dict.keys():
        # 修复循环索引问题
        weighted_params = []
        for i in range(len(client_weight)):
            # 对每个客户端i计算其邻居的加权平均
            neighbor_weights = torch.stack([local_models[j].state_dict()[key] * graph_matrix[i,j] 
                                         for j in range(len(client_weight))], 0).sum(0)
            weighted_params.append(neighbor_weights / graph_matrix.sum(dim=1)[i])
        
        # 合并所有客户端的加权参数
        neighbor_dict[key] = torch.stack(weighted_params, 0).mean(0)
    
    # 融合两层聚合结果
    for key in global_dict.keys():
        global_dict[key] = 0.2 * global_dict[key] + 0.8 * neighbor_dict[key]
        
    global_model.load_state_dict(global_dict)
    return global_model