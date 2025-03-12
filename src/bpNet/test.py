##PG2023
# python test.py -r "2023_07_04_10_17_57-ch_mery-refined" -b "2023_07_15_14_55_22-ch_mery-wColorAug" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "mery" -v "concat" -t True
# python test.py -r "2023_07_06_05_16_01-ch_malcom-refined" -b "2023_07_15_14_55_34-ch_malcolm-wColorAug" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "malcolm" -v "concat" -t True
# python test.py -r "2023_07_11_05_35_35-ch_metahuman-refined" -b "2023_07_15_14_55_40-ch_metahuman-wColorAug" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "metahuman" -v "concat" -t True
# python test.py -r "2023_07_11_08_00_17-ch_piers" -b "2023_07_11_07_59_19-ch_piers" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "piers" -v "concat" -t True
# python test.py -r "2023_07_08_16_35_20-ch_girl" -b "2023_07_11_11_34_31-ch_girl" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "girl" -v "concat" -t True
# python test.py -r "2023_06_21_11_43_19-ch_child" -b "2023_06_22_10_27_32-ch_child-rs" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -i "350000" -d "child" -v "concat" -t True
# python test.py -r "2023_06_21_11_43_19-ch_child" -b "2023_06_27_05_06_07-ch_child" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "child" -v "concat" -t True
# python test.py -r "2023_06_21_11_43_19-ch_child" -b "2023_06_27_05_06_07-ch_child" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -i "250000" -d "child" -v "concat" -t True

##EG2024
    ## Realigned automatic patch version
# python test.py -r "2023_09_06_10_52_50-ch_piers" -b "2023_09_06_10_41_25-ch_piers" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "piers" -v "concat" -t True
# python test.py -r "2023_09_06_10_52_50-ch_piers" -b "2023_09_06_10_41_25-ch_piers" -s "../../dataset/image/real/sihun_brow_test" -ani "../../dataset/animation/real/train_local.pth" -d "piers" -v "concat" -t True

## New eye2lips1 refined
# python test.py -r "2023_09_01_07_00_31-ch_piers" -b "2023_09_01_07_01_26-ch_None" -s "../../dataset/image/real/sihun_brow_test_refined" -ani "../../dataset/animation/real/train_local.pth" -i "250000" -d "piers" -v "concat" -t True
# python test.py -r "2023_09_01_07_00_31-ch_piers" -b "2023_09_01_07_01_26-ch_None" -s "../../dataset/image/real/sihun_brow_test_refined" -c "100000" -ani "../../dataset/animation/real/train_local.pth" -i "250000" -d "piers" -v "concat" -t True
# python test.py -r "2023_09_01_07_00_31-ch_piers" -b "2023_09_01_07_01_26-ch_None" -s "../../dataset/image/real/sihun_brow_test_refined" -c "070000" -ani "../../dataset/animation/real/train_local.pth" -i "250000" -d "piers" -v "concat" -t True
# python test.py -r "2023_09_01_07_00_31-ch_piers" -b "2023_09_01_07_01_26-ch_None" -s "../../dataset/image/real/sihun_brow_test_refined" -c "020000" -ani "../../dataset/animation/real/train_local.pth" -i "250000" -d "piers" -v "concat" -t True

## Netmarble 2024 projcet report
# python test.py -r "2024_01_25_05_12_34-None" -b "2024_01_29_09_48_31-ch_metasihun" -s "../../dataset/image/real/unused/sihun_brow_train_old" -c "140000" -ani "../../dataset/animation/real/train_local.pth" -i "040000" -d "metahuman" -v "concat" -t True

## CGF revision additional experiment
# python test.py -r "2024_06_19_04_33_32-ch_mery" -b "2024_06_19_04_45_23-ch_mery_deformed" -s "../../dataset/image/real/unused/sihun_brow_train_old" -ani "../../dataset/animation/real/train_local.pth" -d "mery_deformed" -v "concat" -t True -n true -ani_t './../../dataset/animation/metahuman/train_local.pth'


# utils
import argparse
import os
import glob
import mediapy as mp
import numpy as np
import torch
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), './../utils/'))
# import char_util as cu
# from json_writer import JSON_writer as jw
# model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../dataset/data_preparation/renderer/')))
from renderer import Renderer
from model_render import ModelBlendshape, ModelPCA
from animation import AnimationPCA

# network
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from network.model import Encoder, Decoder, Autoencoder, Conv, BPNet, MLP

# data
from face_dataset import FaceImgDataset_local
# from face_dataset import FaceImgDataset_local_test

from torch.utils.data import DataLoader
from torch.utils import data

import cv2

import time
from utils import save_image, show_image, torch2numpy, part_combiner, part_resize

def parse():
    parser = argparse.ArgumentParser(description='PyTorch ReenactNet Training')

    parser.add_argument('-r','--ckpt_dir_reenactNet', type=str)
    parser.add_argument('-s','--src_dir', type=str)
    parser.add_argument('-o','--out_dir', type=str, default="./../out")
    parser.add_argument('-ani', '--anim_dir', type=str)
    parser.add_argument('-ani_t', '--trg_anim_dir', type=str, default="./../../dataset/animation/")
    parser.add_argument('-i', '--iter', help="WEM iteration to use", type=str)

    parser.add_argument('-d','--decoder_name', type=str, default="victor")
    parser.add_argument('-b','--ckpt_dir_BPNet', type=str)

    parser.add_argument('-t', '--test_reenact', type=bool, default=False)
    parser.add_argument('-v', '--version', type=str)
    parser.add_argument('-c', '--reenactNet_ckpt', type=str)

    parser.add_argument('-a', '--aug_rotation', type=bool, default=False)

    parser.add_argument('-q', '--cyclic_test', type=bool, default=False)
    parser.add_argument('-e', '--tsne', type=bool, default=False)
    parser.add_argument('-n', '--normalized', type=bool, default=False)

    args = parser.parse_args()
    return args

import torchvision.transforms.functional as TF

import torchvision.transforms as transforms
transform = {
            'base': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
            ])
}

if __name__ == "__main__":

    # set input/output directories
    args = parse()
    ckpt_dir_reenactNet = "../checkpoint/ReenactNet/{0}".format(args.ckpt_dir_reenactNet)
    ### network directory ###
    device = torch.device("cuda:0")

    if args.test_reenact:
        reenactNet_network_dir = os.path.join(ckpt_dir_reenactNet, "train_result")
        network_list = ["eyel", "eyer", "lips"]
        net = {}
        for e in network_list:
            encoder = Encoder().to(device)
            if args.reenactNet_ckpt is not None:
                encoder_path_list = glob.glob(os.path.join(reenactNet_network_dir, f"{e}/encoder_{args.reenactNet_ckpt}.pth"))
            else:    
                encoder_path_list = glob.glob(os.path.join(reenactNet_network_dir, f"{e}/encoder_*.pth"))
            encoder_path_list.sort()
            encoder_path = encoder_path_list[-1]
            encoder.eval()

            decoder = Decoder().to(device)
            if args.reenactNet_ckpt is not None:
                decoder_path_list = glob.glob(os.path.join(reenactNet_network_dir, f"{e}/decoder_{args.decoder_name}_{args.reenactNet_ckpt}.pth"))
            else:
                decoder_path_list = glob.glob(os.path.join(reenactNet_network_dir, f"{e}/decoder_{args.decoder_name}_*.pth"))
            decoder_path_list.sort()
            decoder_path = decoder_path_list[-1]
            decoder.eval()

            print(f"encoder path: {encoder_path}")
            print(f"decoder path: {decoder_path}")

            reenactNet_num = encoder_path.split('_')[-1].replace('.pth',"").zfill(6)
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))

            autoencoder = Autoencoder(encoder, decoder).to(device)
            autoencoder.eval()

            net[e] = {f"autoencoder_{e}": autoencoder}

        autoencoder_eyel = net["eyel"]["autoencoder_eyel"]
        autoencoder_eyer = net["eyer"]["autoencoder_eyer"]
        autoencoder_lips = net["lips"]["autoencoder_lips"]
        

    if args.version == "default" or args.version == "concat": 
    
        ckpt_dir_BPNet = "../checkpoint/BPNet/{0}".format(args.ckpt_dir_BPNet)

        if args.iter:
            bpNet_path = os.path.join(ckpt_dir_BPNet, "BPNet_{}.pth".format(args.iter.zfill(6)))
        else:
            bpNet_path_list = glob.glob(os.path.join(ckpt_dir_BPNet, "BPNet_*.pth"))
            bpNet_path_list.sort()
            bpNet_path = bpNet_path_list[-1]

        bpNet_num = bpNet_path.split('_')[-1].replace('.pth',"").zfill(6)
        out_dir = os.path.join(ckpt_dir_BPNet, f"test_R{reenactNet_num}_B{bpNet_num}")
        print("BPNet bpNet path: ", bpNet_path)
        # import pdb;pdb.set_trace()
        if args.decoder_name == 'metahuman':
            bpNet = BPNet(num_blendshpshapes_or_pcas=40).to(device)
        # else:
        #     bpNet = BPNet().to(device)
        elif args.decoder_name == 'mery_deformed':
            bpNet = BPNet(num_blendshpshapes_or_pcas=40).to(device)
        else:
            bpNet = BPNet().to(device) # CGF revision original 
            # bpNet = BPNet(num_blendshpshapes_or_pcas=51).to(device) # CGF revision blendshape version
            
        bpNet.load_state_dict(torch.load(bpNet_path, map_location=device), strict=False)
        bpNet.eval()

    os.makedirs(out_dir, exist_ok=True)
    out_parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(out_parts_dir, exist_ok=True)
    out_result_dir = os.path.join(out_dir, "result")
    os.makedirs(out_result_dir, exist_ok=True)

    # if args.decoder_name == "victor":
    #     center = torch.tensor([0.0, 0.0, 0.0])
    #     scale = 0.37
    # elif args.decoder_name == "mery":
    #     center = torch.tensor([0.0, -0.5, 0.0])
    #     scale = 0.55
    # elif args.decoder_name == "child": 
    #     center = torch.tensor([0.0, 0.5, 0.0])
    #     scale=0.65
    # elif args.decoder_name == 'malcolm':
    #     center = torch.tensor([0.0, 0.5, 0.0])
    #     scale=0.65
    if args.decoder_name == "victor":
        center_diff = torch.tensor([0.0, 0.0, 0.0])
        scale = 0.37
    elif args.decoder_name == "mery":
        center_diff = torch.tensor([0.0, 0.8, 0.0])
        scale=0.68 # orig: 0.55
    elif args.decoder_name == "mery_deformed":
        center_diff = torch.tensor([0.0, 0.8, 0.0])
        scale=0.68 # orig: 0.55
    # elif args.decoder_name == "mery":   # CGF c2c
    #     center_diff = torch.tensor([0.0, -0.5, 0.0])
    #     scale=0.55
    elif args.decoder_name == "child": 
        center_diff = torch.tensor([0.0, 1.0, 0.0])
        scale=0.7
    elif args.decoder_name == 'malcolm':
        center_diff = torch.tensor([0.0, -2.0, 0.0])
        scale=0.7
    # elif args.decoder_name == 'malcolm':  # CGF c2c
    #     center_diff = torch.tensor([0.0, 0.5, 0.0])
    #     scale=0.65
    elif args.decoder_name == "girl":
        center_diff = torch.tensor([0.0, 4.0, 0.0]) # orig (0,0,0)
        scale = 0.63
    elif args.decoder_name == "piers":
        center_diff = torch.tensor([0.0, 0.6, 0.0])
        scale = 0.58
    elif args.decoder_name == "metahuman" or args.decoder_name == "metasihun":
        omit_face_ids = []
        for i in range(56354,59426):
            # omit_face_ids.append(i * 2)
            # omit_face_ids.append(i * 2 + 1)
            omit_face_ids.append(i)
        # import pdb;pdb.set_trace()
        for elem in list(np.load("./face_indices.npy")): # omit back side of model
            val = int(elem.split('[')[1].split(']')[0])
            omit_face_ids.append(val)

        center_diff = torch.tensor([0.0, 0.0, 0.0])
        scale = 0.35
        
    ## set model
    model_dir = "../../dataset/mesh/{0}/0_{0}".format(args.decoder_name) # PCA
    # model_dir = "../../dataset/mesh/{0}/0_{0}/blendshape".format(args.decoder_name) # bshp
    
    ## YS
    # model_dir = "../../dataset/mesh/{0}/0_{0}".format("metasihun")
    
    ###############
    ## CGF revision
    ###############
    # model = ModelBlendshape(model_dir, center_diff=center_diff, scale=scale) # Blendshape ver
    model = ModelPCA(model_dir, center_diff=center_diff, scale=scale) # PCA ver
    
    ## set renderer
    renderer = Renderer(model, render_size=128) 
    # result_renderer = Renderer(model, render_size=512)
    
    ## load source images info
    dataset = FaceImgDataset_local(args.src_dir, args.anim_dir, is_pca = True, is_test=True, device=device) # PCA
    # dataset = FaceImgDataset_local(args.src_dir, args.anim_dir, is_pca = False, is_test=True, device=device) # bshp

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    ## if blendshape, prints out blenshape name lists and, writes them into json file
    ## !! NOTE !! if PCA, just comment out
    # bs_name_list = model.get_bs_name_list()
    # print("bs_name_list: ", bs_name_list)
    # json_writer = jw(file_name = os.path.join(out_dir, "result.json"), blendshape_name_list=bs_name_list)
    
    video_result = []
    video = []
    video_parts = []
    id = args.decoder_name
    all_vertices = []
    verts = [] # c2c

    start = time.time()

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataset))
        output_parameter_full = torch.tensor([])
        
        if args.normalized: #need to rescale into original PCA parameter range
            trg_animation = AnimationPCA(args.trg_anim_dir, normalize=True)

            min_vals, max_vals = trg_animation.min_vals.detach(), trg_animation.max_vals.detach()
                
        for idx, data in pbar:
            
            ## YS
            # if idx == 571:
            #     break
            if args.test_reenact:
                src_img = data['gt']
                src_eyel_img = data['gt_eyel']
                src_eyer_img = data['gt_eyer']
                src_lips_img = data['gt_lips']

                src_eyel_img_fake, _ = autoencoder_eyel(src_eyel_img, type="eyel")
                src_eyer_img_fake, _ = autoencoder_eyer(src_eyer_img, type="eyer")
                src_lips_img_fake, _ = autoencoder_lips(src_lips_img, type="lips")

            # BPNet
            if args.version == "concat":
                if args.test_reenact:
                        concated_img = torch.cat((src_eyel_img_fake, src_eyer_img_fake, src_lips_img_fake), dim=1)
                output_parameter = bpNet(concated_img, version="concat") # concats three local patch prediction outputs
         
            
            # render_img, _, _ = renderer(output_parameter.detach(), random_augment=False)
            if args.normalized: #need to rescale into original PCA parameter range
                # import pdb;pdb.set_trace()
                output_parameter = output_parameter.cpu()
                output_parameter = (output_parameter * (max_vals - min_vals)) + min_vals     
            
            # render_img, _, meshes = renderer(output_parameter.detach(), random_augment=False)
            if args.decoder_name == "metahuman":
                render_img, _, verts_list = renderer(output_parameter.detach(), omit_face_ids=omit_face_ids, random_augment=False, get_verts=True)
            else:
                render_img, _, verts_list = renderer(output_parameter.detach(), random_augment=False, get_verts=True)
            # result_render_img, _ , _ = result_renderer(output_parameter.detach(), random_augment=False)
            render_img = render_img.permute(0,3,1,2)[:,:3,:,:] # video for concatenated images
            # result_render_img = result_render_img.permute(0,3,1,2)[:,:3,:,:] # video for intact images
            # result_render_img = result_render_img.detach().cpu()
            
            # ## YS ## Vertex error estimation
            # mesh_verts = meshes.verts_packed().unsqueeze(0)
            # all_vertices.append(mesh_verts.cpu().numpy())

            result = torch.cat((src_img.detach().cpu(), render_img.detach().cpu()), dim=3)
            idx_to_str = str(idx).zfill(5)
            # json_writer.append_data(output_parameter[0,:].detach().cpu().numpy().astype(float).tolist())

            if args.test_reenact:
                eyel_img = torch.cat((src_eyel_img.detach().cpu(), src_eyel_img_fake.detach().cpu()), dim=3)
                eyer_img = torch.cat((src_eyer_img.detach().cpu(), src_eyer_img_fake.detach().cpu()), dim=3)
                lips_img = torch.cat((src_lips_img.detach().cpu(), src_lips_img_fake.detach().cpu()), dim=3)
                parts_img = torch.cat((eyel_img, eyer_img, lips_img), dim=2)

                save_image("{0}/{1}.png".format(out_parts_dir, idx_to_str), parts_img)
                # video_parts += [torch2numpy(parts_img).transpose(1, 2, 0)]

            # if show_image(reenact_BP_image):
            #     break
            # output_parameter_full = torch.cat((output_parameter_full, output_parameter.detach().cpu()), dim=0).float()
            save_image("{0}/{1}.png".format(out_dir, idx_to_str), result)

            # video += [torch2numpy(result).transpose(1, 2, 0)]
            # verts.append(verts_list[0])
            # save_image("{0}/{1}_result.png".format(out_result_dir, idx_to_str), result_render_img)
            # video_result += [torch2numpy(result_render_img).transpose(1, 2, 0)]

        # if args.test_reenact:
        #     mp.write_video( "{0}/{1}.mp4".format(out_parts_dir, args.decoder_name), np.stack(video_parts, axis=0) ) # np_video: (frame_num, width, height, channel)
        
        # print("#---------SAVING JSON------------#")
        # json_writer.write_json()
        # print("#---------SAVED JSON------------#")
        
        ## ## YS Vertex error estimation
        # all_vertices_concat = np.concatenate(all_vertices, axis=0)
        # np.savetxt('retargeted.csv', all_vertices_concat.reshape(-1,3))
    
        elapsed_time = time.time() - start
        print(elapsed_time) # 347.42503452301025 이미지 렌더 128, concat해서 저장까지만
        # 83.98550271987915
        breakpoint()
        verts = torch.stack(verts, dim=0)
        np.save(f'{out_dir}/vertices.npy', verts.cpu().numpy())
        mp.write_video( "{0}/{1}.mp4".format(out_dir, args.decoder_name), np.stack(video, axis=0) ) # np_video: (frame_num, width, height, channel)
        mp.write_video( "{0}/{1}_result.mp4".format(out_result_dir, args.decoder_name), np.stack(video_result, axis=0) ) # np_video: (frame_num, width, height, channel)
        # cv2.destroyAllWindows()