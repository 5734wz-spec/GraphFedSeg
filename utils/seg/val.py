import logging
import torch
from medpy.metric.binary import hd95
from scipy import ndimage
import numpy as np
import pdb
from utils.loss_func import Dice_Loss_val
import torch.nn as nn
import torch.nn.functional as F

def val(model, dataloader, args):
    model.eval()

    loss = 0.0
    seg_loss = Dice_Loss_val()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):

            image, label = data['image'], data['label']

            image = image.cuda()
            label = label.cuda()

            if args.fl_method == 'FedEvi':
                logit = model(image)[0]
                logit = torch.clamp_max(logit, 80)
                alpha = torch.exp(logit)+1
                S = torch.sum(alpha, dim=1, keepdim=True) 
                loss += seg_loss(alpha/S, label)
            else:
                pred = model(image)[1]
                loss += seg_loss(pred, label)

        return loss / len(dataloader)

def val(model, dataloader, args, global_model=None):
    model.eval()
    seg_loss = Dice_Loss_val()
    total_score = 0.0
    
    with torch.no_grad():
        # 获取全局模型参数（用于计算协同度）
        global_params = torch.cat([p.view(-1) for p in global_model.parameters()]) if global_model else None
        
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()
            label = data['label'].cuda()
            
            # 前向传播获取预测和模型参数
            pred = model(image)[1]
            local_params = torch.cat([p.view(-1) for p in model.parameters()])
            # ===== 新增置信度监控 =====
            prob = torch.softmax(pred, dim=1)
            confidence = prob.max(dim=1)[0].mean()
            
            # 记录到TensorBoard（需要确保writer和round_idx可用）
            if 'writer' in globals() and 'round_idx' in globals():
                writer.add_scalar(f'client_{client_idx}/confidence', confidence, round_idx)
            # ===== 核心指标计算 =====
            # 基础分割性能（反向指标，值越小越好）
            dice_loss = seg_loss(pred, label)
            
            # 模型协同度（余弦相似度，值越大越好）
            similarity = F.cosine_similarity(global_params.unsqueeze(0), local_params.unsqueeze(0)) if global_model else 1.0
            
            # 综合验证分数 = 分割性能 * 协同度系数
            # 系数设计：0.7基础权重 + 0.3协同度（将相似度从[-1,1]映射到[0,1]）
            collab_coeff = 0.7 + 0.3 * (similarity + 1)/2  # 修改变量名为英文
            batch_score = dice_loss * collab_coeff  # 同步修改变量名
            
            total_score += batch_score.item()
    
    return total_score / len(dataloader)  # 综合得分越低越好