import logging
import torch
from medpy.metric.binary import hd95
from scipy import ndimage
import numpy as np
np.bool = bool
import pdb
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as F
import cv2

# def test(dataset, model, dataloader, client_idx, args, savefig=True):
#     model.eval()

#     if not os.path.exists('visualization/{}/{}/seed{}/client{}/gt/'.format(args.dataset, args.fl_method, args.seed, client_idx+1)):
#         os.makedirs('visualization/{}/{}/seed{}/client{}/gt/'.format(args.dataset, args.fl_method, args.seed, client_idx+1))
#     if not os.path.exists('visualization/{}/{}/seed{}/client{}/pred/'.format(args.dataset, args.fl_method, args.seed, client_idx+1)):
#         os.makedirs('visualization/{}/{}/seed{}/client{}/pred/'.format(args.dataset, args.fl_method, args.seed, client_idx+1))

#     pred_list = torch.tensor([]).cuda()
#     label_list = torch.tensor([]).cuda()

#     iters = 0
#     with torch.no_grad():
#         for _, (_, data) in enumerate(dataloader):
#             image, label = data['image'], data['label']

#             image = image.cuda()
#             label = label.cuda()
#             label_list = torch.cat((label_list, label), 0)

#             logit = model(image)[0]

#             pred_class = torch.argmax(logit, dim=1, keepdim=True)    

#             for i in range(image.shape[0]):
#                 iters += 1
#                 if savefig == True:
#                     plt.imsave('./visualization/{}/{}/seed{}/client{}/pred/pred_{}.png'.format(args.dataset, args.fl_method, args.seed, client_idx+1,iters), pred_class[i,0,...].cpu().numpy(), cmap='gray')
#                     plt.imsave('./visualization/{}/{}/seed{}/client{}/gt/gt_{}.png'.format(args.dataset, args.fl_method, args.seed, client_idx+1,iters), label[i,0,...].cpu().numpy(), cmap='gray')

#                 processed_pred = torch.tensor([]).cuda()
                
#                 for class_idx in range(1, logit.shape[1]):   # 正类
#                     processed_pred = torch.cat((processed_pred, connectivity_region_analysis(pred_class[i:i+1] > class_idx-1)), 1)
                
#                 pred_list = torch.cat((pred_list, processed_pred), 0)

#         pred_list = pred_list.cpu().numpy()
#         label_list = label_list.cpu().numpy()

#         dice_score, hd95_score = cal_dice_hd95(pred_list, label_list)

#         return dice_score, hd95_score



def connectivity_region_analysis(mask):    
    mask_np = mask.cpu().numpy()
    label_im, nb_labels = ndimage.label(mask_np) 
    sizes = ndimage.sum(mask_np, label_im, range(nb_labels + 1))

    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return torch.from_numpy(label_im).cuda()

def cal_dice_hd95(pred, label):
    dice_score = np.zeros(pred.shape[1])
    hd95_score = np.zeros(pred.shape[1])

    smooth = 1e-5
    for data_idx in range(pred.shape[0]):
        avg_sample_dice = 0
        for class_idx in range(pred.shape[1]): 
            label_i = (label>class_idx).astype('float')
            
            # 确保输入数据是有效的二值图像
            label_binary = label_i[data_idx, 0, ...] > 0.5
            pred_binary = pred[data_idx, class_idx, ...] > 0.5
            
            # 计算Dice系数
            inter = (pred_binary * label_binary).sum()
            pred_sum = pred_binary.sum()
            label_sum = label_binary.sum()
            dice_idx = (2*inter + smooth) / (pred_sum + label_sum + smooth)
            dice_score[class_idx] += dice_idx

            # 计算HD95前确保数据是bool类型
            try:
                hd95_idx = hd95(label_binary.astype(bool), pred_binary.astype(bool))
            except Exception as e:
                print(f"HD95计算错误: {e}")
                hd95_idx = 0  # 出错时返回默认值
                
            hd95_score[class_idx] += hd95_idx

    return dice_score / pred.shape[0], hd95_score / pred.shape[0]


def test(dataset, model, dataloader, client_idx, args, savefig=True):
    model.eval()

    if not os.path.exists('visualization/{}/{}/seed{}/client{}/gt/'.format(args.dataset, args.fl_method, args.seed, client_idx+1)):
        os.makedirs('visualization/{}/{}/seed{}/client{}/gt/'.format(args.dataset, args.fl_method, args.seed, client_idx+1))
    if not os.path.exists('visualization/{}/{}/seed{}/client{}/pred/'.format(args.dataset, args.fl_method, args.seed, client_idx+1)):
        os.makedirs('visualization/{}/{}/seed{}/client{}/pred/'.format(args.dataset, args.fl_method, args.seed, client_idx+1))

    pred_list = torch.tensor([]).cuda()
    label_list = torch.tensor([]).cuda()

    iters = 0
    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image, label = data['image'], data['label']
            file_paths = data['file_path']  # 获取文件路径列表
            
            image = image.cuda()
            label = label.cuda()
            label_list = torch.cat((label_list, label), 0)

            logit = model(image)[0]

            pred_class = torch.argmax(logit, dim=1, keepdim=True)    

            for i in range(image.shape[0]):
                iters += 1
                # 从路径中提取原始文件名ID
                file_name = os.path.basename(file_paths[i])  # 获取类似"sample123.npy"
                file_id = file_name.replace("sample", "").split('.')[0]  # 得到"123"

                if savefig:
                    plt.imsave(
                        f'./visualization/{args.dataset}/{args.fl_method}/seed{args.seed}/client{client_idx+1}/pred/pred_{file_id}.png',
                        pred_class[i,0,...].cpu().numpy(),
                        cmap='gray'
                    )
                    plt.imsave(
                        f'./visualization/{args.dataset}/{args.fl_method}/seed{args.seed}/client{client_idx+1}/gt/gt_{file_id}.png',
                        label[i,0,...].cpu().numpy(),
                        cmap='gray'
                    )

                processed_pred = torch.tensor([]).cuda()
                
                for class_idx in range(1, logit.shape[1]):   # 正类
                    processed_pred = torch.cat((processed_pred, connectivity_region_analysis(pred_class[i:i+1] > class_idx-1)), 1)
                
                pred_list = torch.cat((pred_list, processed_pred), 0)

        pred_list = pred_list.cpu().numpy()
        label_list = label_list.cpu().numpy()

        dice_score, hd95_score = cal_dice_hd95(pred_list, label_list)

        return dice_score, hd95_score