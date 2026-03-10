import os
import SimpleITK as sitk
# import nibabel as nib
import os
import numpy as np
from glob import glob
import time
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import copy
import pickle
from natsort import natsorted



def mr_norm(x, r=0.99):
    # normalize mr image
    # x: w,h
    _x = x.flatten().tolist()
    _x.sort()
    vmax = _x[int(len(_x) * r)]
    vmin = _x[0]
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / vmax
    return x


def prepare_polyp(save_dir, ori_dir):
    client_name = ['Kvasir', 'ETIS', 'CVC-ColonDB', 'CVC-ClinicDB']  # 1000, 196, 380，612

    client_data_list = []
    client_mask_list = []
    for i, site_name in enumerate(client_name):
        # client_data_list.append(sorted(glob('{}/{}/image/*'.format(ori_dir, site_name))))
        # client_mask_list.append(sorted(glob('{}/{}/mask/*'.format(ori_dir, site_name))))
        client_data_list.append(natsorted(glob('{}/{}/image/*'.format(ori_dir, site_name))))  # 自然排序
        client_mask_list.append(natsorted(glob('{}/{}/mask/*'.format(ori_dir, site_name)))) 
    for client_idx in range(len(client_name)):
        if not os.path.exists('{}/client{}/data_npy'.format(save_dir, client_idx+1)):
            os.makedirs('{}/client{}/data_npy'.format(save_dir, client_idx+1))

        for data_idx in range(len(client_data_list[client_idx])):
            print('{}_{}'.format(client_idx, data_idx))
            data_path = client_data_list[client_idx][data_idx]
            mask_path = client_mask_list[client_idx][data_idx]

            img = Image.open(data_path)
            if img.size[0] >= img.size[1]:
                W = 384
                H = int((img.size[1]/img.size[0]) * W)
            else:
                H = 384
                W = int((img.size[0]/img.size[1]) * H)


            img = img.resize((W,H), Image.BICUBIC)
            img_np = np.asarray(img)

            PAD_H1 = (384-H) // 2
            if (384-H) % 2 == 0:
                PAD_H2 = PAD_H1
            else:
                PAD_H2 = PAD_H1 + 1

            PAD_W1 = (384-W) // 2
            if (384-W) % 2 == 0:
                PAD_W2 = PAD_W1
            else:
                PAD_W2 = PAD_W1 + 1

            img_np = np.pad(img_np, ((PAD_H1, PAD_H2), (PAD_W1,PAD_W2), (0,0)),constant_values=0)

            mask = Image.open(mask_path)
            mask = mask.resize((W,H), Image.NEAREST)
            mask_np = copy.deepcopy(np.asarray(mask))
            
            if len(mask_np.shape)==2:
                mask_np = np.expand_dims(mask_np, axis=2)
            elif len(mask_np.shape)==3:
                mask_np = np.expand_dims(mask_np[...,0], axis=2)
                
            mask_np = np.pad(mask_np, ((PAD_H1, PAD_H2), (PAD_W1,PAD_W2), (0,0)),constant_values=0)

            mask_np[mask_np<128] = 0
            mask_np[mask_np>128] = 1

            sample = np.dstack((img_np, mask_np))
            np.save('{}/client{}/data_npy/sample{}.npy'.format(save_dir, client_idx+1, data_idx+1), sample)
            print(f"原始文件: {os.path.basename(data_path)} -> 生成文件: sample{data_idx+1}.npy")

def prepare_prostate(save_dir, ori_dir):
    client_name = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']

    for client_idx in range(len(client_name)):
        cnt_sum = 0
        seg_paths = glob(ori_dir + '/' + client_name[client_idx] + '/*segmentation*')
        seg_paths.sort()
        print('[INFO]', client_name[client_idx], len(seg_paths))
        img_paths = [p[:-20] + '.nii.gz' for p in seg_paths]

        for case_idx in range(len(seg_paths)): # case
            itk_image = sitk.ReadImage(img_paths[case_idx])
            itk_mask = sitk.ReadImage(seg_paths[case_idx])
            image = sitk.GetArrayFromImage(itk_image)
            mask = sitk.GetArrayFromImage(itk_mask)

            case_name = img_paths[case_idx].split('/')[-1][:6]

            cnt = np.zeros(2, )
            for slice_idx in range(image.shape[0]):
                slice_name = 'slice_{:03d}.npy'.format(slice_idx)
                slice_image = mr_norm(image[slice_idx]) # 384, 384
                slice_mask = (mask[slice_idx] > 0).astype(int)
                if slice_mask.max() > 0:
                    cnt[1] += 1
                else:
                    continue    
                cnt[0] += 1

                slice_image = np.expand_dims(slice_image,2)

                sample = np.dstack((slice_image, np.expand_dims(slice_mask,2)))
                # 构建完整的保存路径  
                dir_path = os.path.join(save_dir, f'client{client_idx + 1}', 'data_npy', case_name)  
                # 确保目录存在  
                os.makedirs(dir_path, exist_ok=True)  
  
                np.save('{}/client{}/data_npy/{}/{}'.format(save_dir, client_idx+1, case_name, slice_name), sample)

            cnt_sum += cnt[1]
            print('case {}, cnt {}'.format(case_idx, cnt[1]))

        print(cnt_sum)

# def prepare_prostate(save_dir, ori_dir):
#     client_name = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']

#     for client_idx in range(len(client_name)):
#         cnt_sum = 0
#         seg_paths = glob(ori_dir + '/' + client_name[client_idx] + '/*segmentation*')
#         seg_paths.sort()
#         print('[INFO]', client_name[client_idx], len(seg_paths))
#         img_paths = [p[:-20] + '.nii.gz' for p in seg_paths]

#         for case_idx in range(len(seg_paths)):  # case
#             itk_image = sitk.ReadImage(img_paths[case_idx])
#             itk_mask = sitk.ReadImage(seg_paths[case_idx])
#             image = sitk.GetArrayFromImage(itk_image)
#             mask = sitk.GetArrayFromImage(itk_mask)

#             case_name = img_paths[case_idx].split('/')[-1][:6]

#             cnt = np.zeros(2, )
#             for slice_idx in range(image.shape[0]):
#                 slice_name = 'slice_{:03d}.npy'.format(slice_idx)
#                 slice_image = mr_norm(image[slice_idx])  # 384, 384
#                 slice_mask = (mask[slice_idx] > 0).astype(int)
#                 if slice_mask.max() > 0:
#                     cnt[1] += 1
#                 else:
#                     continue
#                 cnt[0] += 1

#                 # 确保slice_image为两通道
#                 if len(slice_image.shape) == 2:  # 如果slice_image是单通道
#                     slice_image = np.stack([slice_image, slice_image], axis=-1)  # 复制单通道为两通道
#                 elif slice_image.shape[-1] == 1:  # 如果slice_image已经是单通道的三维数组
#                     slice_image = np.concatenate([slice_image, slice_image], axis=-1)  # 复制单通道为两通道
#                 elif slice_image.shape[-1] > 2:
#                     slice_image = slice_image[..., :2]  # 截取前两个通道

#                 # 确保slice_mask为单通道
#                 if len(slice_mask.shape) == 2:
#                     slice_mask = np.expand_dims(slice_mask, axis=-1)

#                 sample = np.dstack((slice_image, slice_mask))

#                 # 构建完整的保存路径
#                 dir_path = os.path.join(save_dir, f'client{client_idx + 1}', 'data_npy', case_name)
#                 # 确保目录存在
#                 os.makedirs(dir_path, exist_ok=True)

#                 np.save(os.path.join(dir_path, slice_name), sample)

#             cnt_sum += cnt[1]
#             print('case {}, cnt {}'.format(case_idx, cnt[1]))

#         print(cnt_sum)
                
    


def prepare_fundus(save_dir, ori_dir):
    client_name = ['Drishti_GS', 'RIM-ONE-r3','REFUGE_Zeiss', 'REFUGE_Canon', 'BinRushed', 'Margabia']
    client_data_list = []
    client_mask_list = []

    for client_idx in range(len(client_name)):
        if client_idx < 4:
            post_str = ''
        else:
            post_str = '/*/*'
        client_data_list.append(sorted(glob('{}/{}/image/*{}'.format(ori_dir, client_name[client_idx], post_str))))
        client_mask_list.append(sorted(glob('{}/{}/mask/*{}'.format(ori_dir, client_name[client_idx],post_str))))

    
    for client_idx in range(len(client_name)):
        cnt = 0

        if not os.path.exists('{}/{}/data_npy'.format(save_dir, client_name[client_idx])):
            os.makedirs('{}/{}/data_npy'.format(save_dir, client_name[client_idx]))

        for data_idx in range(len(client_data_list[client_idx])):
            cnt += 1

            data_path = client_data_list[client_idx][data_idx]
            mask_path = client_mask_list[client_idx][data_idx]
            print(data_path, mask_path)

            img = Image.open(data_path)
            img = img.resize((384,384), Image.BICUBIC)
            img_np = np.asarray(img)

            # process the mask
            mask = Image.open(mask_path)
            mask = mask.resize((384,384), Image.NEAREST)
            mask_np = np.asarray(mask)

            try:
                mask_np = np.expand_dims(copy.deepcopy(mask_np[...,0]), axis=2)
            except:
                mask_np = np.expand_dims(copy.deepcopy(mask_np), axis=2)

            if client_idx < 4:
                mask_np[mask_np==0] = 2 # optic cup
                mask_np[mask_np==128] = 1   # optic disc
                mask_np[mask_np==255] = 0   # background
            else:
                mask_np[mask_np==128] = 2 # optic cup
                mask_np[mask_np==255] = 1   # optic disc
                mask_np[mask_np==0] = 0   # background

            sample = np.dstack((img_np, mask_np))
            
            np.save('{}/client{}/data_npy/sample{}.npy'.format(save_dir, client_idx+1, cnt), sample)

    
prepare_polyp(save_dir='/opt/data/private/wz/1/data/Polyp_npy', ori_dir='/opt/data/private/wz/DATA/Polyp')
# prepare_prostate('/opt/data/private/wz/DATA/Prostate_npy', ori_dir='/opt/data/private/wz/1/Dataset/Prostate')
# prepare_fundus(save_dir='/opt/data/private/wz/DATA/Fundus_npy', ori_dir='/opt/data/private/wz/1/Dataset/Fundus')
