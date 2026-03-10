import numpy as np
import torch
import time
import pdb
import torch
import copy
import cvxpy as cp
import logging


def scoring_func(global_model, local_model, dataloader, client_idx, args):
    global_model.eval()
    local_model.eval()

    udis_list = torch.tensor([]).cuda()
    udata_list = torch.tensor([]).cuda()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()

            # surrogate global model
            g_logit = global_model(image)[0]
            g_logit = torch.clamp_max(g_logit, 80)
            alpha = torch.exp(g_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True) # batch_size, 1, patch_size, patch_size

            g_pred = alpha / total_alpha
            g_entropy = torch.sum(- g_pred * torch.log(g_pred), dim=1)     
            g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            
            g_u_dis = g_entropy - g_u_data
            udis_list = torch.cat((udis_list, g_u_dis.mean(dim=[1,2])))

            # local model
            l_logit = local_model(image)[0]
            l_logit = torch.clamp_max(l_logit, 80)
            alpha = torch.exp(l_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True) # batch_size, 1, patch_size, patch_size
            l_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)

            udata_list = torch.cat((udata_list, l_u_data.mean(dim=[1,2])))

    return udis_list.mean().cpu().numpy(), udata_list.mean().cpu().numpy()




    
    # 计算模型在本地测试数据上的个性化和泛化准确率
    # 参数:
    # model (torch.nn.Module): 要评估的模型。
    # dataloader (torch.utils.data.DataLoader): 测试数据加载器。
    # data_distribution (np.ndarray): 数据分布。

    # 返回:
    # float: 个性化准确率。
    # float: 泛化准确率。
def compute_local_test_accuracy(model, dataloader, data_distribution):

    model.eval()

    toatl_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution).sum()
    
    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total

    

def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]]
        for key in model_i:
            # 确保dw字典中有该键
            if key not in dw[index_clientid[i]]:
                dw[index_clientid[i]][key] = torch.zeros_like(model_i[key])
            
            # 计算参数差异
            if key in initial_global_parameters:
                dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]
            else:
                # 对于不存在的键，使用本地模型参数作为差异
                dw[index_clientid[i]][key] = model_i[key]
    
    # 计算余弦相似度
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            # 使用所有层的参数计算相似度
            if similarity_matric == "all":
                diff = -torch.nn.functional.cosine_similarity(
                    weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0),
                    weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0)
                )
                if diff < -0.9:
                    diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            
            # 也可以添加特定层的相似度计算（如解码器部分）
            elif similarity_matric == "decoder":  # 新增选项
                decoder_keys = [k for k in dw[index_clientid[i]] if 'convu' in k]  # UNet解码器部分
                diff = -torch.nn.functional.cosine_similarity(
                    torch.cat([dw[index_clientid[i]][k].reshape(-1) for k in decoder_keys]).unsqueeze(0),
                    torch.cat([dw[index_clientid[j]][k].reshape(-1) for k in decoder_keys]).unsqueeze(0)
                )
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    return model_similarity_matrix

    
#


#最新修改图更新函数
def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, 
                                dw, fed_avg_freqs, lambda_1, similarity_matric, 
                                uncertainty_scores=None, uncertainty_weight=0.4):
    
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric)
    
    # ========== 新增不确定性融合逻辑 ==========
    if uncertainty_scores is not None:
        # 计算客户端间不确定性相似度矩阵
        uncertainty_sim = 1 / (1 + np.abs(
            uncertainty_scores[:, None] - uncertainty_scores[None, :]
        ))
        
        # 将原始图矩阵与不确定性相似度融合（使用加权平均）
        graph_matrix = graph_matrix.numpy()
        graph_matrix = 0.9 * graph_matrix + 0.1 * uncertainty_sim  # 可调整融合比例
        
    
        # 归一化处理
        np.fill_diagonal(graph_matrix, 0)
        graph_matrix = graph_matrix / graph_matrix.sum(axis=1, keepdims=True)
        graph_matrix = torch.from_numpy(graph_matrix).float()

    # 将融合后的矩阵传入优化函数
    graph_matrix = optimizing_graph_matrix_neighbor(
        graph_matrix, 
        index_clientid,
        model_difference_matrix,
        lambda_1,
        fed_avg_freqs
    )
    return graph_matrix






def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    if isinstance(fed_avg_freqs, dict):
        p = np.array(list(fed_avg_freqs.values()))
    elif isinstance(fed_avg_freqs, np.ndarray):
        p = fed_avg_freqs
    else:
        raise ValueError("fed_avg_freqs must be either a dictionary or a numpy.ndarray")
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()

        graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
    return graph_matrix
  



# 扁平化模型参数（所有层）
def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params



#  基于图矩阵进行模型聚合。

    # 参数:
    # cfg: 配置对象。
    # graph_matrix (torch.Tensor): 图矩阵。
    # nets_this_round (dict): 本轮参与的客户端模型字典。
    # global_w (dict): 全局模型参数。
    # cluster_models (dict): 聚类模型字典

#
def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w, cluster_models):
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    client_num = cfg["client_num"]
    
    # 初始化临时参数容器
    for client_id in range(client_num):
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

    # 参数聚合核心逻辑
    for client_id in range(client_num):
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        # 参数聚合
        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id]  # 直接使用传入的state_dict
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

        # 聚类向量生成
        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id]
            net_para_flatten = weight_flatten_all(net_para)
            if torch.linalg.norm(net_para_flatten) != 0:
                cluster_model_state += net_para_flatten * (
                    aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para_flatten)
                )

    # 更新客户端模型参数
    for client_id in range(client_num):
        cluster_models[client_id].load_state_dict(tmp_client_state_dict[client_id])
    
    return cluster_model_vectors


def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)


    # 计算模型在测试数据上的损失。

    # 参数:
    # net (torch.nn.Module): 要评估的模型。
    # test_data_loader (torch.utils.data.DataLoader): 测试数据加载器。

    # 返回:
    # float: 损失。
def compute_loss(net, test_data_loader):
    net.eval()
    loss, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data+_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            loss += torch.nn.functional.cross_entropy(out, target).item()
            total += x.data.size()[0]
    net.to('cpu')
    return loss / float(total)





