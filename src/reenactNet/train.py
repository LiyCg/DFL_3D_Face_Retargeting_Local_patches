## PG2023 - reviving
## child
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/child/local_child_refined" -v "ch_child" -ch "child" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train" "../../dataset/image/child/local_child" -v "ch_child" -ch "child" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train" "../../dataset/image/child/local_child" -d "2023_07_19_09_01_27-ch_child" -r "120000" -v "ch_child" -ch "child" --save_ckpt
    ## child resume train
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/child/local_child_refined" -v "ch_child" -ch "child" -d "2023_06_21_11_43_19-ch_child" -r "030000" --save_ckpt 
## mery
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/mery/local_mery_refined" -v "ch_mery" -ch "mery" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train" "../../dataset/image/mery/local_mery" -v "ch_mery" -ch "mery" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train" "../../dataset/image/mery/local_mery" -d "2023_07_19_09_05_31-ch_mery" -r "120000" -v "ch_mery" -ch "mery" --save_ckpt

## malcolm
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/malcolm/local_malcolm_refined" -v "ch_malcom" -ch "malcolm" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train" "../../dataset/image/malcolm/local_malcolm" -v "ch_malcom" -ch "malcolm" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train" "../../dataset/image/malcolm/local_malcolm" -d "2023_07_19_09_21_22-ch_malcom" -r "120000" -v "ch_malcom" -ch "malcolm" --save_ckpt
      
## girl(refined newly created dataset)
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/girl/local_girl_refined_v2" -v "ch_girl" -ch "girl" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/girl/local_girl_refined_v3" -v "ch_girl" -ch "girl" --save_ckpt (more neutral expressions)
    ## girl
# python train.py -s "../../dataset/image/real/sihun_brow_train" "../../dataset/image/girl/local_girl" -v "ch_girl" -ch "girl" --save_ckpt 
    ## girl (resume train with iter for .pth file specified) 
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/girl/local_girl_refined_v3" -v "ch_girl" -ch "girl" -d "2023_07_20_10_45_00-ch_girl" -r "040000" --save_ckpt 

## piers
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/piers/local_piers_refined" -v "ch_piers" -ch "piers" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/piers/local_piers_refined" -v "ch_piers" -ch "piers" --save_ckpt

## metahuman
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/metahuman/local_metahuman_refined" -v "ch_metahuman" -ch "metahuman" --save_ckpt

## CGF revision
    ## train with more closed eye and neutral face sihun images 
# python train.py -s "../../dataset/image/real/sihun_train_2408_brow_neutClosedAdded" "../../dataset/image/child/local_child_WEM_19215" -v "ch_child" -ch "child" --save_ckpt
# python train.py -s "../../dataset/image/real/local_real_original_woRot" "../../dataset/image/child/local_child_WEM_19215" -v "ch_child" -ch "child" --save_ckpt
    ## train with extremely deformed mery for showing extreme overlapping patch cases
# python train.py -s "../../dataset/image/real/unused/sihun_brow_train_old" "../../dataset/image/mery_deformed/local_mery_deformed" -v "ch_mery" -ch "mery" --save_ckpt

## CGF 2nd revision
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/metahuman/local_metahuman_refined" -v "ch_metahuman" -ch "metahuman" --save_ckpt


# utils
import argparse
import random
import time
import torch
from torch import nn
import torchvision
import os
# import wandb
import sys
import glob
import numpy as np
from utils import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../utils')))
# from loggers import get_logger, print_options

# data 
from face_dataset import FaceImgDataset
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms

# network
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from network.model import Encoder, Decoder, Autoencoder

# optimization
import torch.optim as optim
from pytorch_msssim import ssim #, ms_ssim, SSIM, MS_SSIM

# from torchsummary import summary as summary
import torch.nn.functional as F

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
    times = "_".join(times)

    parser = argparse.ArgumentParser(description='PyTorch new Training')

    parser.add_argument('-s','--src_data_dirs', metavar='N', type=str, nargs='+', help='multiple path of src dataset')
    parser.add_argument('-t','--trg_data_dir', type=str, default="/todo")
    parser.add_argument('-ch','--character_name', type=str, help='model for render scale')
    parser.add_argument('-b','--save_ckpt', action='store_true')
    parser.add_argument('-p','--project_name', type=str, default="YAT")

    parser.add_argument('-d','--load_ckpt_name', type=str, help="resume train from checkpoint")
    parser.add_argument('-r','--load_iter', type=str, help="resume train from the last iter")

    parser.add_argument('-v', '--version', type=str, \
                        choices=['ch_child','ch_girl','ch_piers','ch_malcolm','ch_mery','ch_metahuman',\
                            'c2c_m2g', 'c2c_g2m', 'c2c_mal2me', 'c2c_me2mal', 'ch_Victor', 'ch_Emily', 'ch_Morphy', 'ch_VMan', 'ch_Malcolm'], 
                        help='name for reenactNet version')
    parser.add_argument('-c','--ckpt_dir', type=str, default="../checkpoint/ReenactNet")
    parser.add_argument('-lr','--lr', default=1e-5, type=float)

    parser.add_argument('--lambda_idt',      type=float, default=10.0,   help='weight for idt loss')
    parser.add_argument('--lambda_SSIM',      type=float, default=1.0,   help='weight for SSIM loss')
    parser.add_argument('--iters',      type=float, default=200000,   help='iterations')
    args = parser.parse_args()

    if args.load_ckpt_name:
        args.ckpt_dir = os.path.join("../checkpoint/ReenactNet", args.load_ckpt_name) # 2022_12_03_00_56_52-L1_ssim_aug_color_random

    else:
        args.ckpt_dir = os.path.join(args.ckpt_dir, f"{times}-{args.version}")
        os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # log_path         = args.ckpt_dir+'/reenactNet_{}.log'
    # train_log        = get_logger("{}".format(times), path=log_path)
    # train_log.info("\n"+print_options(args, parser))
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

def train(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.project_name == "YAT":
        ids = ['source', 'target']
    else:
        ids = ['real', args.character_name]

    ############################
    # experiment configuration #
    ############################
    is_test = False
    
    aug_random = False
    aug_affine_color = True
    aug_color = False
    
    batch_size = 16
    ############################
    
    # default
    datasets = [FaceImgDataset(src_data_dir, is_test=is_test, aug_random=aug_random, aug_affine_color=aug_affine_color, aug_color=aug_color, id=id, device=device) for (src_data_dir, id) in zip(args.src_data_dirs, ids) ]
    dataloaders = [sample_data(DataLoader(dataset, batch_size=batch_size, sampler=data_sampler(dataset, shuffle=True), num_workers=4, drop_last=True)) for dataset in datasets]

    ckpt_dir_sub = os.path.join(args.ckpt_dir, "train_result")
    ckpt_dirs = {} 
    net = {}

    src_param=[]
    trg_param=[]

    parts = ["eyel", "eyer", "lips"]

    for part in parts:
        encoder  = Encoder().to(device)
        decoders = [Decoder(part).to(device) for i in range(len(dataloaders))]

        if args.load_ckpt_name:
            encoder.load_state_dict(torch.load(os.path.join(ckpt_dir_sub, f"{part}/encoder_{args.load_iter}.pth")))
            for i, decoder in enumerate(decoders):
                decoder.load_state_dict(torch.load(os.path.join(ckpt_dir_sub, f"{part}/decoder_{ids[i]}_{args.load_iter}.pth")))
                print(f"load {part} decoder_{ids[i]}_{args.load_iter}.pth")

        autoencoders = [Autoencoder(encoder, decoder) for decoder in decoders]

        net[part] = {f"autoencoders_{part}": autoencoders, f"encoder_{part}": encoder, f"decoders_{part}": decoders}
        src_param+=list(autoencoders[0].parameters())
        trg_param+=list(autoencoders[1].parameters())
        ckpt_dirs[part] = os.path.join(ckpt_dir_sub, part)

    # eyel / eyer / lips 
    for ckpt_dir in ckpt_dirs.values():
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

    autoencoders_eyel = net["eyel"]["autoencoders_eyel"]
    autoencoders_eyer = net["eyer"]["autoencoders_eyer"]
    autoencoders_lips = net["lips"]["autoencoders_lips"]
    optimizers = [optim.Adam(net_param, lr=args.lr, betas=(0.9, 0.999)) for net_param in [src_param, trg_param]]

    criterion_idt = nn.L1Loss()
    criterion_ssim = ssim

    from_iter = 0
    if args.load_iter:
        from_iter = int(args.load_iter)
    
    start_time = time.time()
    times = []
    for i in range(from_iter, args.iters + 1):

        for dataloader, autoencoder_eyel, autoencoder_eyer, autoencoder_lips, optimizer, id \
            in zip(dataloaders, autoencoders_eyel, autoencoders_eyer, autoencoders_lips, optimizers, ids):

            optimizer.zero_grad()
            
            data_ = next(dataloader)

            gt_eyel_img, src_eyel_img = data_['gt_eyel'], data_['s_eyel']
            gt_eyer_img, src_eyer_img = data_['gt_eyer'], data_['s_eyer']
            gt_lips_img, src_lips_img = data_['gt_lips'], data_['s_lips']

            gt_eyel_img = gt_eyel_img.permute(0,3,1,2).to(device, dtype=torch.float)
            src_eyel_img = src_eyel_img.permute(0,3,1,2).to(device, dtype=torch.float)

            gt_eyer_img = gt_eyer_img.permute(0,3,1,2).to(device, dtype=torch.float)
            src_eyer_img = src_eyer_img.permute(0,3,1,2).to(device, dtype=torch.float)
            
            gt_lips_img = gt_lips_img.permute(0,3,1,2).to(device, dtype=torch.float)
            src_lips_img = src_lips_img.permute(0,3,1,2).to(device, dtype=torch.float)
        
            src_eyel_img_fake, _ = autoencoder_eyel(src_eyel_img, type="eyel")
            src_eyer_img_fake, _ = autoencoder_eyer(src_eyer_img, type="eyer")
            src_lips_img_fake, _ = autoencoder_lips(src_lips_img, type="lips")

            loss_eyel = criterion_idt(src_eyel_img_fake, gt_eyel_img)
            loss_eyer = criterion_idt(src_eyer_img_fake, gt_eyer_img)
            loss_lips = criterion_idt(src_lips_img_fake, gt_lips_img)

            loss_idt = loss_eyel + loss_eyer + loss_lips 

            loss_ssim_eyel = (1 - criterion_ssim(src_eyel_img_fake, gt_eyel_img, data_range=1, size_average=True))
            loss_ssim_eyer = (1 - criterion_ssim(src_eyer_img_fake, gt_eyer_img, data_range=1, size_average=True))
            loss_ssim_lips = (1 - criterion_ssim(src_lips_img_fake, gt_lips_img, data_range=1, size_average=True))
            
            loss_ssim = loss_ssim_eyel + loss_ssim_eyer + loss_ssim_lips

            loss = loss_idt*args.lambda_idt + loss_ssim*args.lambda_SSIM

            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                idx_to_str = str(i).zfill(6)
                
                if id != args.character_name:
                    with torch.no_grad():
                        # import pdb; pdb.set_trace()
                        result_eyel, _ = autoencoders_eyel[1](gt_eyel_img,type="eyel")
                        torchvision.utils.save_image(
                            torch.cat([gt_eyel_img, result_eyel], dim=2), 
                            f"{ckpt_dirs['eyel']}/src_2_trg_{idx_to_str}.jpg", 
                            normalize=True, 
                            value_range=(0,1)
                        )
                        result_eyer, _ = autoencoders_eyer[1](gt_eyer_img,type="eyer")
                        torchvision.utils.save_image(
                            torch.cat([gt_eyer_img, result_eyer], dim=2), 
                            f"{ckpt_dirs['eyer']}/src_2_trg_{idx_to_str}.jpg", 
                            normalize=True, 
                            value_range=(0,1)
                        )
                        result_lips, _ = autoencoders_lips[1](gt_lips_img,type="lips")
                        torchvision.utils.save_image(
                            torch.cat([gt_lips_img, result_lips], dim=2), 
                            f"{ckpt_dirs['lips']}/src_2_trg_{idx_to_str}.jpg", 
                            normalize=True, 
                            value_range=(0,1)
                        )
                
                torchvision.utils.save_image(
                        torch.cat([src_eyel_img, gt_eyel_img, src_eyel_img_fake], dim=2), 
                        "{0}/result_{1}_{2}.jpg".format(ckpt_dirs['eyel'], id, idx_to_str), 
                        normalize=True, 
                        value_range=(0,1)
                    )
                torchvision.utils.save_image(
                        torch.cat([src_eyer_img, gt_eyer_img, src_eyer_img_fake], dim=2), 
                        "{0}/result_{1}_{2}.jpg".format(ckpt_dirs['eyer'], id, idx_to_str), 
                        normalize=True, 
                        value_range=(0,1)
                    )
                torchvision.utils.save_image(
                        torch.cat([src_lips_img, gt_lips_img, src_lips_img_fake], dim=2), 
                        "{0}/result_{1}_{2}.jpg".format(ckpt_dirs['lips'], id, idx_to_str), 
                        normalize=True, 
                        value_range=(0,1)
                    )
        
        if args.save_ckpt:
            idx_to_str = str(i).zfill(6)
            
            if i % 500 == 0:
                print("{0} \ttotal loss: {1:.4f} \tloss_eyel: {2:.4f} \tloss_eyer: {3:.4f} \tloss_lips: {4:.4f} \tloss_ssim_eyel: {5:.4f} \tloss_ssim_eyer: {6:.4f} \tloss_ssim_lips: {7:.4f}".format(i, loss.item(), \
                    loss_eyel.item(), loss_eyer.item(), loss_lips.item(), loss_ssim_eyel.item(), loss_ssim_eyer.item(), loss_ssim_lips.item()))

            if i % 10000 == 0 and i != 0:
                time_elapsed = time.time() - start_time
                times.append(i)
                times.append(time_elapsed)
                torch.save(net['eyel']['encoder_eyel'].state_dict(),  "{0}/encoder_{1}.pth".format(ckpt_dirs['eyel'], idx_to_str))
                torch.save(net['eyer']['encoder_eyer'].state_dict(),  "{0}/encoder_{1}.pth".format(ckpt_dirs['eyer'], idx_to_str))
                torch.save(net['lips']['encoder_lips'].state_dict(),  "{0}/encoder_{1}.pth".format(ckpt_dirs['lips'], idx_to_str))

                [torch.save(decoder_eyel.state_dict(), "{0}/decoder_{1}_{2}.pth".format(ckpt_dirs['eyel'], src_data_dir.split("/")[-2], idx_to_str)) \
                        for src_data_dir, decoder_eyel in zip(args.src_data_dirs, net['eyel']['decoders_eyel'])]
                [torch.save(decoder_eyer.state_dict(), "{0}/decoder_{1}_{2}.pth".format(ckpt_dirs['eyer'], src_data_dir.split("/")[-2], idx_to_str)) \
                    for src_data_dir, decoder_eyer in zip(args.src_data_dirs, net['eyer']['decoders_eyer'])]
                [torch.save(decoder_lips.state_dict(), "{0}/decoder_{1}_{2}.pth".format(ckpt_dirs['lips'], src_data_dir.split("/")[-2], idx_to_str)) \
                    for src_data_dir, decoder_lips in zip(args.src_data_dirs, net['lips']['decoders_lips'])]
        time_check_txt = "./time_check.txt"
        with open(time_check_txt, "w") as file:
            for t in times:
                file.write(str(t) + "\n")   
    

if __name__ == '__main__':

    args = parse()
    train(args)

    
