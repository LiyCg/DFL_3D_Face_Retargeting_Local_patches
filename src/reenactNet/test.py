"""
Visualize and test if expression transfer works

    : A(real) encoder -> B(stylized) decoder 
    
-> only outputs concat images(3 patches) 

"""

# python test.py -r "../checkpoint/ReenactNet/2023_05_11_13_18_09-ch_girl" -s "../../dataset/image/real/sihun_brow_test" -a "../../dataset/animation/val_normal.pth" -d "girl" -c "150000"

## CGF revision additional experiment
# python test.py -r "../checkpoint/ReenactNet/2024_06_19_04_33_32-ch_mery" -s "../../dataset/image/real/unused/sihun_brow_test_old" -a "../../dataset/animation/real/train_local.pth" -o "./out/mery_deformed/" -d "mery_deformed" -c "200000"

### CGF 2nd revision
# python test.py -r "../checkpoint/ReenactNet/2024_09_17_06_27_51-ch_metahuman" -s "../../dataset/image/real/sihun_brow_test" -a "../../dataset/animation/real/train_local.pth" -o "./out/mery_deformed/" -d "metahuman" -c "200000"

### ICCV 2025 comparison experiment
# python test.py -r "../checkpoint/ReenactNet/2025_02_23_11_14_23-ch_Emily" -s "../../dataset/Your_Avatar_Talks/data/images/test/result_mead/inf_sets/mead_0_hallo" -a "../../dataset/animation/real/train_local.pth" -o "./out/Emily_mead_0_hallo/" -d "target" -c "050000"
# python test.py -r "../checkpoint/ReenactNet/2025_02_23_10_57_22-ch_Victor" -s "../../dataset/Your_Avatar_Talks/data/images/test/result_mead/inf_sets/mead_0_hallo" -a "../../dataset/animation/real/train_local.pth" -o "./out/Victor_mead_0_hallo/" -d "target" -c "050000"


import argparse
import os
import sys
import glob
import mediapy as mp
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from face_dataset import FaceImgDataset_local

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../train/network/')))
from model import Encoder, Decoder, Autoencoder

def parse():
    parser = argparse.ArgumentParser(description='PyTorch ReenactNet Testing')
    parser.add_argument('-r', '--ckpt_dir_reenactNet', type=str)
    parser.add_argument('-s', '--src_dir', type=str)
    parser.add_argument('-a', '--anim_dir', type=str)
    parser.add_argument('-o', '--out_dir', type=str, default="./out")
    parser.add_argument('-d','--decoder_name', type=str, default="victor")
    parser.add_argument('-c', '--reenactNet_ckpt_iter', type=str)
    args = parser.parse_args()
    return args

# def save_image(out_dir,file_name, img):
#     img = img.squeeze().cpu().numpy().transpose(1, 2, 0)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join(out_dir,file_name), (img * 255).astype(np.uint8))

def save_image(img_name, img):
    img = img * 255
    pic = img.detach().cpu().numpy()
    result_pic = pic[0, :, :, :]
    for i in range(1, pic.shape[0]):
        result_pic = np.concatenate((result_pic, pic[i, :, :, :]), axis=1)
    result_pic = result_pic.transpose(1,2,0)
    result_pic = cv2.cvtColor(result_pic, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_name, result_pic)

def test(args):
    
    device = torch.device("cuda:0")

    os.makedirs(args.out_dir, exist_ok=True)
    
    
    reenactNet_network_dir = os.path.join(args.ckpt_dir_reenactNet, "train_result")
    network_list = ["eyel", "eyer", "lips"]
    net = {}
    for e in network_list:
        encoder = Encoder().to(device)
        
        encoder_path_list = glob.glob(os.path.join(reenactNet_network_dir, f"{e}/encoder_{args.reenactNet_ckpt_iter}.pth"))
        # import pdb;pdb.set_trace()
        encoder_path_list.sort()
        encoder_path = encoder_path_list[-1]
        encoder.eval()

        decoder = Decoder().to(device)
        
        decoder_path_list = glob.glob(os.path.join(reenactNet_network_dir, f"{e}/decoder_{args.decoder_name}_{args.reenactNet_ckpt_iter}.pth"))
        
        decoder_path = decoder_path_list[-1]
        decoder.eval()

        print(f"encoder path: {encoder_path}")
        print(f"decoder path: {decoder_path}")

        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        autoencoder = Autoencoder(encoder, decoder).to(device)
        autoencoder.eval()

        net[e] = {f"autoencoder_{e}": autoencoder}

    autoencoder_eyel = net["eyel"]["autoencoder_eyel"]
    autoencoder_eyer = net["eyer"]["autoencoder_eyer"]
    autoencoder_lips = net["lips"]["autoencoder_lips"]

    # Load the dataset
    dataset = FaceImgDataset_local(args.src_dir, args.anim_dir, is_test=True, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Process dataset
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataset))
        for idx, data in pbar:
            # src_img = data['gt']
            src_eyel_img = data['gt_eyel']
            src_eyer_img = data['gt_eyer']
            src_lips_img = data['gt_lips']

            src_eyel_img_fake, _ = autoencoder_eyel(src_eyel_img, type="eyel")
            src_eyer_img_fake, _ = autoencoder_eyer(src_eyer_img, type="eyer")
            src_lips_img_fake, _ = autoencoder_lips(src_lips_img, type="lips")

            eyel_img = torch.cat((src_eyel_img.detach().cpu(), src_eyel_img_fake.detach().cpu()), dim=3)
            eyer_img = torch.cat((src_eyer_img.detach().cpu(), src_eyer_img_fake.detach().cpu()), dim=3)
            lips_img = torch.cat((src_lips_img.detach().cpu(), src_lips_img_fake.detach().cpu()), dim=3)
            
            parts_img = torch.cat((eyel_img, eyer_img, lips_img), dim=2)
            
            file_path = "{}.png".format(idx)
            save_image(os.path.join(args.out_dir, file_path), parts_img)


def create_video(data_dir, video_name):
    # Create a list of image file names
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])

    # Read the first image to get the shape
    first_image = cv2.imread(os.path.join(data_dir, image_files[0]))

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    # We specify output file name (eg: output.mp4), codec code ('mp4v'), number of frames per second (fps) and frame size
    height, width, layers = first_image.shape
    video = cv2.VideoWriter(f'{data_dir}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for image_file in image_files:
        # Reading each image and writing it to the video frame
        image_frame = cv2.imread(os.path.join(data_dir, image_file))
        video.write(image_frame)

    # After all frames are written, close the output video file.
    video.release()
    print(f"DONE making {data_dir}/{video_name}.mp4")


if __name__ == "__main__":
    
    ## make results into images
    args = parse()
    test(args)
    
    ## images into MP4 vid
    # create_video("../../train/reenactNet/out", "reenact_sihun_girl")
    
    