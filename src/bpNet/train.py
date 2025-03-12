
# utils 
import argparse
import time
import torch
from torch import nn
import torchvision
import os
import sys 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import glob
# import wandb

# from utils.loggers import get_logger, print_options


# model 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../dataset/data_preparation/renderer/')))
from renderer import Renderer
from model_render import ModelPCA
from animation import AnimationBlendShape, AnimationPCA

# data
from face_dataset import FaceImgDataset_local
from torch.utils.data import DataLoader
from torch.utils import data

# network
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from network.model import BPNet, MLP

ngpu= 1 
# num_epochs = 8
lr = 5e-5 # 원래 이거

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9
beta2 = 0.999
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
workers = 0
batch_size = 16

def parse():
    now = time.localtime()
    times = map(str, 
        [
            now.tm_year, 
            str(now.tm_mon).zfill(2), 
            str(now.tm_mday).zfill(2), 
            str(now.tm_hour).zfill(2), 
            str(now.tm_min).zfill(2), 
            str(now.tm_sec).zfill(2)
        ]
    )

    parser = argparse.ArgumentParser(description='PyTorch BPNetwork Training')

    parser.add_argument('-d','--root_dir', type=str, help='path of dataset')
    parser.add_argument('-a','--anim_dir', type=str, default="/todo")
    parser.add_argument('-ch','--char_name', type=str)
    parser.add_argument('-p','--project_name', type=str, default="YAT")

    parser.add_argument('-k','--save_ckpt', action='store_true')
    parser.add_argument('-b','--ckpt_BPNet_dir', type=str)

    parser.add_argument('-c','--ckpt_dir', type=str, default="../checkpoint/BPNet")
    parser.add_argument('-ck','--load_ckpt_name', type=str, help="resume train from checkpoint")
    parser.add_argument('-r','--load_iter', type=str, help="resume train from the last iter")

    parser.add_argument('-v', '--version', type=str, choices=['train_BPNet','concat_aug_color', 'train_WEM'])

    args = parser.parse_args()
    if args.load_ckpt_name:
        args.ckpt_BPNet_dir = args.load_ckpt_name
    else:
        args.ckpt_BPNet_dir = "../checkpoint/BPNet/{0}-ch_{1}".format("_".join(times), args.char_name)
        os.makedirs(args.ckpt_BPNet_dir,exist_ok=True)

    return args

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def data_sampler(dataset, shuffle, distributed=False):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

if __name__ == '__main__':
    args = parse()
    # wandb.init(project="Local-Retargeting-PG", entity="kaist-vml")
    # wandb.run.name = args.version
    
    # ## CGF revision pca param version (with normalization)
    # uncomment for use
    if args.anim_dir.split('.')[-1] == 'pth' or args.anim_dir.split('.')[-1] == 'pt': # takes .pt for YAT
        if args.version == 'concat_aug_color':
            face_dataset = FaceImgDataset_local(args.root_dir, args.anim_dir, is_pca=True, is_test=False, aug_color=True, device=device)
        else:
            face_dataset = FaceImgDataset_local(args.root_dir, args.anim_dir, is_pca=True, is_test=False, device=device)
        
        if args.project_name == "YAT": ## if YAT
            if args.char_name == 'Emily':
                num_basis = 121
            elif args.char_name == 'Malcolm':
                num_basis = 32
            elif args.char_name == 'Morphy':
                num_basis = 46
            elif args.char_name == 'Victor':
                num_basis = 45
            elif args.char_name == 'VMan':
                num_basis = 101      
            bpNet = BPNet(num_blendshpshapes_or_pcas=num_basis).to(device)
        else: # if everything else (till YAT)
            if args.char_name == "metahuman" or args.char_name == 'metasihun':
                bpNet = BPNet(num_blendshpshapes_or_pcas=40).to(device)
            elif args.char_name == "mery_deformed":
                bpNet = BPNet(num_blendshpshapes_or_pcas=40).to(device)
            else: 
                bpNet = BPNet(num_blendshpshapes_or_pcas=20).to(device)
    else:
        if args.version == 'concat_aug_color':
            face_dataset = FaceImgDataset_local(args.root_dir, args.anim_dir, is_pca=False, is_test=False, aug_color=True, device=device)
        else:
            face_dataset = FaceImgDataset_local(args.root_dir, args.anim_dir, is_pca=False, is_test=False, device=device)
        # ## CGF revision bshp param version
        ## uncomment for use
        bpNet = BPNet(num_blendshpshapes_or_pcas=51).to(device)
    
    # dataloader = sample_data(DataLoader(face_dataset, batch_size=batch_size, sampler=data_sampler(face_dataset, shuffle=True), num_workers=8, drop_last=True)) # 초기화 전에 worker 프로세스가 메모리를 잡아버려서 오류가 발생
    dataloader = sample_data(DataLoader(face_dataset, batch_size=batch_size, sampler=data_sampler(face_dataset, shuffle=True), num_workers=workers, drop_last=True))
    
    if args.load_ckpt_name:
        args.ckpt_BPNet_dir = os.path.join(args.ckpt_dir, args.load_ckpt_name)
        bpNet.load_state_dict(torch.load(os.path.join(args.ckpt_BPNet_dir, f"BPNet_{args.load_iter}.pth")))
        print(f"load BPNet_{args.load_iter}.pth")
    else:
        ckpt_dir_eyel = os.path.join(args.ckpt_BPNet_dir, "eyel")
        ckpt_dir_eyer = os.path.join(args.ckpt_BPNet_dir, "eyer")
        ckpt_dir_lips = os.path.join(args.ckpt_BPNet_dir, "lips")
        os.makedirs(ckpt_dir_eyel, exist_ok=True)
        os.makedirs(ckpt_dir_eyer, exist_ok=True)
        os.makedirs(ckpt_dir_lips, exist_ok=True)
    
    optimizer = optim.Adam(list(bpNet.parameters()), lr=lr, betas=(beta1, beta2))

    criterion = nn.L1Loss()

    print("-----------train begin-----------")

    from_iter = 0
    if args.load_iter:
        from_iter = int(args.load_iter)

    start_time = time.time()
    times = []
    for i in range(from_iter,500000 + 1):

        idx_to_str = str(i).zfill(6)
        optimizer.zero_grad()
        # ===================forward=====================
        data_ = next(dataloader)
        eyel_aug_img = data_['s_eyel'].to(device) # warped input batch of images
        eyer_aug_img = data_['s_eyer'].to(device) # warped input batch of images 
        lips_aug_img = data_['s_lips'].to(device) # warped input batch of images

        eyel_aug_img = eyel_aug_img.permute(0,3,1,2).to(device, dtype=torch.float)/255.0
        eyer_aug_img = eyer_aug_img.permute(0,3,1,2).to(device, dtype=torch.float)/255.0
        lips_aug_img = lips_aug_img.permute(0,3,1,2).to(device, dtype=torch.float)/255.0

        concated_img = torch.cat((eyel_aug_img, eyer_aug_img, lips_aug_img), dim=1)
        output_parameter = bpNet(concated_img, version="concat")

        gt_parameter = data_['p'] # train data image parameters
        gt_parameter = (gt_parameter).to(device, dtype=torch.float)
        loss_param = criterion(gt_parameter, output_parameter)

        loss = loss_param 

        loss.backward()
        optimizer.step()

        if args.save_ckpt:
            if i % 50 == 0: # 500 to 250 
                data_ = next(dataloader)
                with torch.no_grad(): # validation
                    eyel_aug_img_val = data_['v_s_eyel'].to(device)
                    eyer_aug_img_val = data_['v_s_eyer'].to(device)
                    lips_aug_img_val = data_['v_s_lips'].to(device)
                    eyel_aug_img_val = eyel_aug_img_val.permute(0,3,1,2).to(device, dtype=torch.float)/255.0
                    eyer_aug_img_val = eyer_aug_img_val.permute(0,3,1,2).to(device, dtype=torch.float)/255.0
                    lips_aug_img_val = lips_aug_img_val.permute(0,3,1,2).to(device, dtype=torch.float)/255.0
                    concated_img = torch.cat((eyel_aug_img_val, eyer_aug_img_val, lips_aug_img_val), dim=1)
                    bpNet.eval()
                    output_parameter_val = bpNet(concated_img, version="concat")
                    gt_parameter_val = data_['v_p']
                    gt_parameter_val = (gt_parameter_val).to(device, dtype=torch.float)
                    val_loss = criterion(gt_parameter_val, output_parameter_val).detach()

                print("{0} \ttotal loss: {1:.4f} \tval loss: {2:.4f}".format(i, loss.item(), val_loss.item()))
                
                if i == 0:
                    torchvision.utils.save_image(
                             torch.cat([eyel_aug_img, eyer_aug_img, lips_aug_img], dim=2), 
                            "{0}/aug_{1}.jpg".format(args.ckpt_BPNet_dir, idx_to_str), 
                            normalize=True, 
                            value_range=(0,1)
                        )
                
            if i % 2500 == 0 and i != 0:
                time_elapsed = time.time() - start_time
                times.append(i)
                times.append(time_elapsed)
                torch.save(bpNet.state_dict(), "{0}/BPNet_{1}.pth".format(args.ckpt_BPNet_dir, idx_to_str))
                time_check_txt = "./time_check.txt"
                with open(time_check_txt, "w") as file:
                    for t in times:
                        file.write(str(t) + "\n") 

