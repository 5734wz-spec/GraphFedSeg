import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import pdb
import logging


class Dice_Loss_val(nn.Module):
    def __init__(self):
        super(Dice_Loss_val, self).__init__()
        self.smooth = 1e-5

    def forward(self, pred, label):
        K = pred.shape[1]
        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)

        dice_score = 0.0
        
        for class_idx in range(K):   
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = pred[:,class_idx,...].sum() + one_hot_y[:,class_idx,...].sum()

            dice_score += (2*inter + self.smooth) / (union + self.smooth)

        loss = 1 - dice_score/K

        return loss
    

def kl_divergence(alpha):
    shape = list(alpha.shape)
    shape[0] = 1
    
    # import pdb
    # pdb.set_trace()
    # print(shape)
    ones = torch.ones(tuple(shape)).cuda()

    S = torch.sum(alpha, dim=1, keepdim=True) 
    # pdb.set_trace()
    first_term = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(S))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl.mean()    


    
class Train_Dice_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-5

    def forward(self, pred, label):
        K = pred.shape[1]
        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)

        dice_score = 0.0
        
        for class_idx in range(K):   
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = (pred[:,class_idx,...]**2).sum() + one_hot_y[:,class_idx,...].sum()  # 保持平方项
            dice_score += (2*inter + self.smooth) / (union + self.smooth)

        return 1 - dice_score/K



class GraphAwareLoss(nn.Module):
    def __init__(self, lambda_graph=0.1, lambda_uncertainty=0.3):
        super().__init__()
        self.base_loss = Train_Dice_Loss()
        self.lambda_graph = lambda_graph
        self.lambda_uncertainty = lambda_uncertainty

    def forward(self, pred, label, current_model, neighbor_models, graph_weights, uncertainty):
        """
        新增参数说明：
        neighbor_models: 当前客户端的邻居模型列表（根据协作图选择）
        graph_weights: 协作权重向量（来自graph_matrix的对应行）
        """
        # 基础分割损失
        dice_loss = self.base_loss(pred, label)
        
        # ===== 图结构正则项（基于邻居模型）=====
        neighbor_sim = 0.0
        current_params = torch.cat([p.view(-1) for p in current_model.parameters()])
        
        # 计算与每个邻居的相似度加权和
        for i, neighbor in enumerate(neighbor_models):
            neighbor_params = torch.cat([p.view(-1) for p in neighbor.parameters()])
            sim = torch.cosine_similarity(current_params, neighbor_params, dim=0)
            neighbor_sim += graph_weights[i] * sim
            
        # 正则项 = 1 - 加权相似度均值（鼓励模型与邻居相似）
        graph_penalty = 1 - neighbor_sim / (torch.sum(graph_weights) + 1e-6)
        
        # ===== 改进的不确定性正则项 =====
        uncertainty_penalty = torch.log(1 + uncertainty**2)
        
        # 组合损失
        total_loss = dice_loss + \
                   self.lambda_graph * graph_penalty + \
                   self.lambda_uncertainty * uncertainty_penalty
        
        return total_loss


class EDL_UncertaintyLoss(nn.Module):
    def __init__(self, annealing_step=10):
        super().__init__()
        self.annealing_step = annealing_step
        
    def forward(self, logit, label, epoch_num):
        # 证据计算
        evidence = torch.exp(torch.clamp(logit, -10, 10))
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # ===== 不确定性感知Dice损失 =====
        dice_loss = 0
        for class_idx in range(logit.shape[1]):
            pred = alpha[:,class_idx,...] / S.squeeze(1)
            inter = (pred * label[:,class_idx,...]).sum()
            union = (pred**2).sum() + label[:,class_idx,...].sum()
            dice_loss += (2*inter + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice_loss/logit.shape[1]
        
        # ===== 不确定性正则项 =====
        uncertainty = (logit.shape[1] / S).mean()  # 基于证据的不确定性
        kl_loss = kl_divergence(alpha) * min(1.0, epoch_num/self.annealing_step)
        
        return dice_loss + 0.1*kl_loss + 0.2*uncertainty