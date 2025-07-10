import torch
import numpy as np
import cv2
import os

from models.Blender_complex import BlendedModel
from models.Generator_new import GenModel

import torchvision.utils as vutils


def save_concat_image(tensors, path):
    tensors = [t.detach().cpu() for t in tensors]
    grid = vutils.make_grid(tensors, nrow=len(tensors), normalize=False, scale_each=False)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()
    cv2.imwrite(path, ndarr) 


def load_clear_image(path, size=(640, 480)):
    img = cv2.imread(path)
    img = img.astype(np.float32) / 255.0
    img = img[8:-8, 8:-8]  # crop border
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) 
    return img


def load_depth_image(path, size=(640, 480)):
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    dep = dep[8:-8, 8:-8]  # crop border
    dep = dep.astype(np.float32) / 1000.0
    dep = cv2.resize(dep, size, interpolation=cv2.INTER_LINEAR)
    dep = 0.5 + (dep - dep.min()) / (dep.max() - dep.min()) * 4.5 # set the depth range here
    dep = torch.from_numpy(dep).unsqueeze(0).unsqueeze(0) 
    return dep


def generate_synthetic(clear_path, depth_path, real_path, G_weight_path, B_weight_path, output_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    clear = load_clear_image(clear_path).to(device)
    real = load_clear_image(real_path).to(device)
    depth = load_depth_image(depth_path).to(device)
    netG = GenModel().to(device)
    netB = BlendedModel().to(device)
    netG.load_state_dict(torch.load(G_weight_path, map_location=device))
    netB.load_state_dict(torch.load(B_weight_path, map_location=device))
    netG.eval()
    netB.eval()

    noise = torch.randn(1, 1, 10).to(device)

    with torch.no_grad():
        fake_noise = netG(noise)
        fake_img, smooth, smooth_forward = netB(clear, depth, fake_noise)

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, 'synthetic.png')
    save_concat_image([clear[0], smooth[0], smooth_forward[0], fake_img[0], real[0]], save_path)
    #  in-air, smooth, smooth with forward scattering, full synthetic image with noise, underwater image
    print(f"Synthetic image saved to: {save_path}")


if __name__ == '__main__':
    mode = 'lake'  # select the mode here, lake or easi

    output_dir = 'outs'                # output path
    clear_img_path = 'imgs/clear.png'  # in-air image path
    depth_img_path = 'imgs/depth.png'  # depth image path

    if mode == 'lake':
        weight_G_dir = 'weights/Lake_G_simple.pt'  # Noise generator path
        weight_B_dir = 'weights/Lake_B_simple.pt'  # Blender path
        real_image_path = 'imgs/Lake.jpg'          # underwater image (just for demo)
    elif mode == 'easi':
        weight_G_dir = 'weights/EASI_G_simple.pt'
        weight_B_dir = 'weights/EASI_B_simple.pt'
        real_image_path = 'imgs/EASI.png'
    else:
        raise ValueError("Mode must be 'lake' or 'easi'.")

    generate_synthetic(clear_img_path, depth_img_path, real_image_path, weight_G_dir, weight_B_dir, output_dir)
