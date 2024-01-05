"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
import time

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from thop import profile

from data_RGB import get_test_data
#from MPRNet import MPRNet#,SWALNet
#from SWALNet import WaveletModel
#from MSGN1G import MSGN
#from STRN import STRN#STRN_woALL
#from STRN_64_10 import STRN#STRN_woALL
# from COformer82_test import COformer#STRN_woALL,STRN_64_10_RB3
# from COformer79 import COformer#STRN_woALL,STRN_64_10_RB3
from DAWN_v3g import DAWN#STRN_woALL,STRN_64_10_RB3
#from WD2N11 import WD2N#STRN_woALL,STRN_dehazing
#from STRN_plus1 import STRN#STRN_woALL,STRN_lowlight
#from PCNet1 import PCNet#STRN_woALL

from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using MSGN')

#parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results', type=str, help='Directory for results')#results,lowlight\LOL1000,dehazing
#parser.add_argument('--result_dir', default='G:/demo/raindrop/', type=str, help='Directory for results')#results,lowlight\LOL1000,dehazing
#parser.add_argument('--weights', default='./pretrained_models/model_best.pth', type=str, help='Path to weights')#model_epoch_350, model_latest,model_deraining
parser.add_argument('--weights', default='./checkpoints/Deraining/models/DAWN_v3g/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')#STRN_woALL,model_deraining

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus #args.gpus

model_restoration = DAWN()  #,MSGN,STRN,MPRNet,WD2N,WaveletModel,DAWN

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)

file = 'par_gflops.txt'
# device_ids = [0,1]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if len(device_ids)>1:
    # model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)
    # model_restoration.to(device)	
model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

#datasets = ['lowlight\EXDARK', 'lowlight\LOL1000', 'lowlight\TEST148', 'lowlight\VOC144', 'dehazing\RESIDE', 'dehazing\NYU']
#datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'BDD350', 'COCO350', 'Real127', 'Real200']#,'RID', 'RIS']
#datasets = ['lowlight\TEST148'],['rainhaze\cityscape']
# datasets = ['Real127', 'Real200', 'RID', 'RIS']#'RID', 'RIS'
#datasets = ['RIS']#'RID', 'RIS'
#datasets = ['realrain','realsnow']#'RID', 'RIS'
# datasets = ['Rain100L', 'Rain100H', 'Test100', 'Real127', 'Test1200']#, 'Test1200', 'Real127']#, 'BDD350', 'COCO350', 'Real127'], 'Real200'
# datasets = ['Rain100L', 'Rain100H', 'Test100']
# datasets = ['Test20']
datasets = ['LOL15']#'BDD350', 'COCO350'
# datasets = ['Raindrop']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.result_dir, dataset, 'input')# elformer, mprnet, label, input
    #print("rgb_dir_test",rgb_dir_test)
    if (rgb_dir_test):
        print("rgb_dir_test",rgb_dir_test)
        print('***************************')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)


    result_dir  = os.path.join(args.result_dir, dataset, 'DAWN_v3g_best')#COformer8_best,COformer74_best_visual5_BTR
	#RN_64_10_best,STRN_best_att_fea2,STRN_best_mask2
    #result_dir  = args.result_dir
    if not os.path.exists(result_dir):
        utils.mkdir(result_dir)
    all_time =0
    count = 0
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            input_    = data_test[0].cuda()

            filenames = data_test[1]
            st_time=time.time()
            restored = model_restoration(input_)
            ed_time=time.time()
            cost_time=ed_time-st_time
            all_time +=cost_time
            count +=1
            #print('spent {} s.'.format(cost_time))
            #print(filenames)
            restored = torch.clamp(restored,0,1)
            # restored = torch.clamp(restored[0],0,1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
    print('spent {} s.'.format(all_time))
    print('spent {} s per item.'.format(all_time/(count)))#(ii+1)
input_test    = input_.cuda()
# print('model_restoration', model_restoration.device)
flops, params = profile(model_restoration, (input_,))
print('flops: ', flops, 'params: ', params)

format_str = ('flops: %.4f, params: %.4f, per_time: %.4f')
a = str(format_str % (flops, params, all_time/(count)))
PSNR_file = open(file, 'a+')
PSNR_file.write(a)
PSNR_file.write('\n')
PSNR_file.close()