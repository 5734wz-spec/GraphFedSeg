import numpy as np
import argparse
import os
import time
import random
import logging
import sys
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from model.unet2d import Unet2D as Model
from data.dataset import generate_dataset
from utils.fed_merge import FedAvg, FedUpdate,FedAvg2
from utils.utils import scoring_func
from utils.utils import *
from utils.seg.val import val
from torchvision.transforms import Lambda
from utils.seg.test import test

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,  default='Polyp', help='dataset')
parser.add_argument('--fl_method', type=str,  default='Graph+nodouble', help='federated method')
parser.add_argument('--max_round', type=int,  default=250, help='maximum round number to train')
parser.add_argument('--max_epoch', type=int,  default=2, help='maximum epoch number to train')
parser.add_argument('--norm', type=str,  default='bn', help='normalization type')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=5e-4, help='  learning rate')
parser.add_argument('--deterministic', type=bool,  default=False, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=3, help='random seed')
# parser.add_argument('--kl_weight', type=float, default=0.01, help='edl kl weight')
parser.add_argument('--ratio', type=float, default=1.0, help='ratio')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
# parser.add_argument('--annealing_step', type=int, default=10, help='annealing_step')
parser.add_argument('--num_classes', type=int, default=2, help='class num')

# ... 原有参数 ...
parser.add_argument('--alpha', type=float, default=0.7, help='协作图超参数')
parser.add_argument('--lam', type=float, default=0.01, help='余弦相似度权重')
parser.add_argument('--difference_measure', type=str, default='all', help='模型差异度量方式')

#修改
parser.add_argument('--uncertainty_weight', type=float, default=0.4, help='不确定性惩罚项权重')
parser.add_argument('--lambda_graph', type=float, default=0.1, help='图结构正则化系数')
# 在原有参数基础上添加
parser.add_argument('--graph_fusion_ratio', type=float, default=0.9, help='图结构与不确定性融合比例')
args = parser.parse_args()
# 在参数解析后添加cfg配置（约第60行）
# ... 原有参数解析代码 ...
# args = parser.parse_args()

# # 新增配置字典
# cfg = {
#     "comm_round": args.max_round,
#     "client_num": client_num,  # 注意这个变量需要在dataset判断后初始化
#     "model_name": "Unet2D",
#     "classes_size": args.num_classes
# }

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)


if __name__ == '__main__':
    # log
    localtime = time.localtime(time.time())
    ticks = '{:>02d}{:>02d}{:>02d}{:>02d}{:>02d}'.format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min, localtime.tm_sec)

    # snapshot_path = "UNet_{}/{}_{}_{}/".format(args.dataset.lower(), args.dataset, args.fl_method, ticks)
    # # 修改原第57行
    # # snapshot_path = "UNet_{}/{}_{}_{}_uw{}/".format(args.dataset.lower(),args.dataset,args.fl_method,ticks,args.uncertainty_weight)

    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    # if not os.path.exists(snapshot_path + '/model'):
    #     os.makedirs(snapshot_path + '/model')

    # logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    snapshot_path = "UNet_{}/{}_{}_{}/".format(args.dataset.lower(), args.dataset, args.fl_method, ticks)
    
    # 使用递归创建目录（自动创建所有不存在的父目录）                    
    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs(os.path.join(snapshot_path, 'model'), exist_ok=True)

    # 修改日志文件路径写法
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),  # 使用标准路径拼接
        # ... 其他参数保持不变 ...
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    

    # init
    dataset = args.dataset
    assert dataset in ['Polyp']
    # assert dataset in ['Fundus']
    # assert dataset in ['Prostate']
    fl_method = args.fl_method
    # assert fl_method in ['pFedGraph']
    assert fl_method in ['Graph+nodouble']
   

    batch_size = args.batch_size
    base_lr = args.base_lr
    max_round = args.max_round
    norm = args.norm
    

    if fl_method == 'FedEvi':
        from utils.seg.train_fedevi import train_fedevi
        bn = False
        norm = 'in'
        val_batch_size = 1

    

    
    if dataset == 'Polyp':
        c = 3
        client_num = 4


    if dataset == 'Prostate':
        c = 3
        client_num = 6

    if dataset == 'Fundus':
        c = 3
        client_num = 6
    

    # cfg["client_num"] = client_num

    # 替换FedEvi相关初始化
    if fl_method == 'Graph+nodouble':
        from utils.seg.train_fedevi import train_graph
        # bn = True
        # norm = 'bn'
        # val_batch_size = batch_size
        bn = False
        norm = 'in'
        val_batch_size = 1
        


    # random seed
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    

    # local dataloader, model, optimizer


    local_train_loaders = []
    local_val_loaders = []
    local_test_loaders = []

    train_num = []

    for client_idx in range(client_num):
        # data
        data_train, data_val, data_test = generate_dataset(dataset=dataset, fl_method=fl_method, client_idx=client_idx)
        train_num.append(len(data_train))



        # dataloader
        train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset=data_val, batch_size=val_batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
                
        local_train_loaders.append(train_loader)
        local_val_loaders.append(val_loader)
        local_test_loaders.append(test_loader)

        # model
       

    writer = SummaryWriter(snapshot_path+'/log')

    result_after_avg = np.zeros(client_num)

    best_val = 9999 # val loss
    best_dice = 0.0

    # global model
    global_model = Model(c=c, num_classes=args.num_classes, norm=norm).cuda()
    global_parameters = global_model.state_dict()
    cluster_model_vectors = {}
    cluster_models = []
    dw = []
    # 在初始化部分添加客户端最佳记录（约第200行附近）
    best_client_dice = {i: 0.0 for i in range(client_num)}
    best_client_models = {i: None for i in range(client_num)}
    # 新增不确定性记录变量
    uncertainty_scores = torch.zeros(client_num).cuda()  # 各客户端不确定性得分

    for i in range(client_num):
        model = Model(c=c, num_classes=args.num_classes, norm=norm).cuda()
        cluster_models.append(model)
        # dw.append({key : torch.zeros_like(value) for key, value in global_models[i].named_parameters()})
        dw.append({k: torch.zeros_like(v) for k, v in global_parameters.items()})
    # 初始化协作图矩阵，对角线元素为 0，非对角线元素相等。
    
    graph_matrix = torch.ones(client_num, client_num) / (client_num-1)
    graph_matrix.fill_diagonal_(0)


    # 本地模型初始化
    local_models = []
    for _ in range(client_num):
        model = Model(c=c, num_classes=args.num_classes, norm=norm).cuda()
        model.load_state_dict(global_parameters)  # 确保使用相同的参数结构
        local_models.append(model)

      # 初始化聚类向量（新增）
    cluster_model_vectors = {
        idx: torch.cat([p.detach().view(-1) for p in model.parameters()])
        for idx, model in enumerate(local_models)
    }


    # 将全局模型的参数加载到本地模型和聚类模型中。
    for net in local_models:
        net.load_state_dict(global_parameters)
    for net in cluster_models:   
        net.load_state_dict(global_parameters)

    local_optimizers = []
    local_schedulers = []
    for client_idx in range(client_num):
        optimizer = torch.optim.Adam(local_models[client_idx].parameters(), lr=args.base_lr, betas=(0.9, 0.99), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)


        local_optimizers.append(optimizer)
        local_schedulers.append(scheduler)

    client_weight = train_num / np.sum(train_num)

    with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
        print('train num: {}'.format(train_num), file=f)
        print('init weight: {}'.format(client_weight), file=f)


    u_dis = np.zeros((client_num, max_round))
    u_data = np.zeros((client_num, max_round))
    
    for round_idx in range(max_round):
        local_models = FedUpdate(global_model, local_models, bn=bn)   # distribute
        
        # 新增：保存初始模型参数（用于后续计算模型差异）

        # nets_param_start = {k: copy.deepcopy(local_models[k]) for k in range(client_num)}
        nets_param_start = {k: copy.deepcopy(local_models[k].state_dict()) for k in range(client_num)}



# ========== 修改训练过程调用方式 ==========
        for client_idx in range(client_num):
            print(client_idx)
            k_neighbors = 3  # 可根据需要调整邻居数量
            neighbor_indices = torch.topk(graph_matrix[client_idx], k=k_neighbors).indices.tolist()
            # 修改为接收模型和不确定性
            local_model, client_uncert = train_graph(
                round_idx=round_idx+1,
                client_idx=client_idx, 
                model=local_models[client_idx],
                local_models=local_models,
                graph_matrix=graph_matrix,
                global_model=global_model,
                dataloader=local_train_loaders[client_idx],
                optimizer=local_optimizers[client_idx],
                args=args,
                cluster_vectors=cluster_model_vectors,  # 新增参数
                neighbor_indices=neighbor_indices  # 新增参数
            )
            local_models[client_idx] = local_model
            uncertainty_scores[client_idx] = client_uncert  # 存储不确定性

        # 替换原有的client_weight使用
        adjusted_weights = client_weight * (1 / (uncertainty_scores.cpu().numpy() + 1e-6))
        adjusted_weights /= adjusted_weights.sum()  # 归一化处理

        






        # update the global model, while the local models haven't been updated
        # global_model = FedAvg(global_model, local_models, client_weight, bn=bn) 
        global_model = FedAvg2(global_model, local_models, client_weight, graph_matrix,bn=bn) 

        model_difference_matrix = cal_model_cosine_difference(
            {k: local_models[k].state_dict() for k in range(client_num)},
            global_parameters,  # 使用全局模型state_dict
            dw,
            "all"
        )

        

       #修改图更新函数调用 
        graph_matrix = update_graph_matrix_neighbor(
        graph_matrix,
        {k: local_models[k].state_dict() for k in range(client_num)},
        global_model.state_dict(),
        dw,
        adjusted_weights,
        args.alpha,
        args.difference_measure,
        uncertainty_scores=uncertainty_scores.cpu().numpy(),  # 新增不确定性参数
        uncertainty_weight=args.uncertainty_weight
    )
      

        # cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters) 
        cluster_model_vectors = aggregation_by_graph(
            cfg={
                "client_num": client_num,  # 从主程序获取
                "comm_round": args.max_round,  # 从命令行参数获取
                "model_name": "Unet2D",  # 硬编码模型名称
                "classes_size": args.num_classes  # 从命令行参数获取
            }, 
            graph_matrix=graph_matrix,
            nets_this_round={k: v.state_dict() for k, v in enumerate(local_models)},
            global_w=global_parameters,
            cluster_models=cluster_models
        )
        # 改进为结合图结构和不确定性的调整方式# 获取当前客户端在图结构中的连接强度（行求和）
        connection_strength = graph_matrix.sum(dim=1).cpu().numpy()
        adjusted_weights = (args.alpha * connection_strength + (1 - args.alpha) * (1 - uncertainty_scores.cpu().numpy()))
        # 与初始客户端数据量权重融合
        adjusted_weights = client_weight * adjusted_weights
        adjusted_weights /= adjusted_weights.sum()
     

        # ========== 修改验证逻辑 ==========
                # ========== 修改验证逻辑 ==========
        local_val_losses = []
        local_dice_scores = []
        local_hd95_scores = []
        
        # 遍历所有客户端进行验证
        for client_idx in range(client_num):
            # 只验证本地个性化模型
            model = local_models[client_idx]
            
            # 获取验证损失
            val_loss = val(model, local_val_loaders[client_idx], args)
            local_val_losses.append(val_loss)
            
            # 获取测试指标
            dice, hd95 = test(dataset, model, local_test_loaders[client_idx], client_idx, args)
            local_dice_scores.append(dice.mean())
            local_hd95_scores.append(hd95.mean())
            
            # 实时打印训练信息
            print(f'Round {round_idx+1} | Client {client_idx} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Dice: {dice[0]:.3f} | HD95: {hd95[0]:.3f}')
            
            # 记录到TensorBoard
            writer.add_scalars(f'client_{client_idx}/val', {
                'loss': val_loss,
                'dice': dice.mean(),
                'hd95': hd95.mean()
            }, round_idx)
            
        # 更新客户端最佳模型（实时保存历史最佳）
        for idx in range(client_num):
            current_dice = local_dice_scores[idx]
            if current_dice > best_client_dice[idx]:
                best_client_dice[idx] = current_dice
                best_client_models[idx] = copy.deepcopy(local_models[idx].state_dict())
                # 保存时会覆盖之前的最佳模型（固定文件名）
                save_path = os.path.join(snapshot_path, f'model/client_{idx}_best.pth')
                torch.save(best_client_models[idx], save_path)
      
                # 更新最佳模型逻辑
        avg_dice = np.mean(local_dice_scores)
        if avg_dice > best_dice:
            best_dice = avg_dice
            with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                print(f'\nFL round {round_idx+1}', file=f)
                print(f'Average Dice: {avg_dice:.4f}', file=f)
                print('Client Details:', file=f)
                for idx in range(client_num):
                    print(f'  Client {idx} => '
                          f'Dice: {local_dice_scores[idx]:.4f} | '
                          f'HD95: {local_hd95_scores[idx]:.4f}', file=f)
                print('='*50, file=f)
            
            # 保存全局最佳模型（原有逻辑保留）
            torch.save(global_model.state_dict(), os.path.join(snapshot_path, 'model/best_global.pth'))
        # 将日志记录合并到单个with块中
        with open(os.path.join(snapshot_path, 'training_log.txt'), 'a') as f:
            # 记录客户端详情
            print(f'Round {round_idx+1}', file=f)
            print('Client Details:', file=f)
            for idx in range(client_num):  # 修复作用域问题
                print(f'  Client {idx} => '
                      f'Dice: {local_dice_scores[idx]:.4f} | '
                      f'HD95: {local_hd95_scores[idx]:.4f}', file=f)
            print('-'*50, file=f)
            
            # 记录全局统计
            print(f'Validation Losses: {[f"{l:.4f}" for l in local_val_losses]}', file=f)
            print(f'Dice Scores: {[f"{s:.3f}" for s in local_dice_scores]}', file=f) 
            print(f'HD95 Scores: {[f"{s:.3f}" for s in local_hd95_scores]}', file=f)
            print('-'*50, file=f)

# ... 后续代码保持不变 ...     
    

    # 保存最终各客户端最佳模型
    for idx in range(client_num):
            current_dice = local_dice_scores[idx]
            if current_dice > best_client_dice[idx]:
                best_client_dice[idx] = current_dice
                best_client_models[idx] = copy.deepcopy(local_models[idx].state_dict())
                # 保存客户端最佳模型（固定文件名）
                save_path = os.path.join(snapshot_path, f'model/client_{idx}_best.pth')
                torch.save(best_client_models[idx], save_path)


            
    # 在训练结束后添加总结（在writer.close()之前添加）
    with open(os.path.join(snapshot_path, 'best_results.txt'), 'a') as f:
        print('\n=== Client Best Summary ===', file=f)
        for idx in range(client_num):
            print(f'Client {idx} Best Dice: {best_client_dice[idx]:.4f}', file=f)
        print('='*30, file=f)
    writer.close()


    

