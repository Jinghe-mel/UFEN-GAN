import torch
import os
import random
from argparse import ArgumentParser as AP
import torch.optim as optim

from models.Blender_complex import BlendedModel
from models.Generator_new import GenModel
from models.Discriminator import DiscModel
from models.Discriminator_mul import DiscModel_mul

from train_scripts.Dataset_rgbd import SimpleDataset
from train_scripts.GAN_train_first_step import first_gan_train
from train_scripts.GAN_train_second_step import second_gan_train


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def set_paras(net, train_bool=True):
    for para in net.parameters():
        para.requires_grad = train_bool


def create_folders(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    first_path = os.path.join(base_path, 'first')
    second_path = os.path.join(base_path, 'second')
    img_path = os.path.join(base_path, 'first_img')
    img_path2 = os.path.join(base_path, 'second_img')
    for path in [first_path, second_path, img_path, img_path2]:
        if not os.path.exists(path):
            os.makedirs(path)
    return first_path, second_path


if __name__ == '__main__':
    ap = AP()

    # === Path and I/O arguments ===
    ap.add_argument('--clear_data', type=str, required=True,help='Path to clear (in-air) images')
    ap.add_argument('--turbid_data', type=str, required=True,help='Path to turbid (underwater) images')
    ap.add_argument('--save_path', default='results/', type=str, help='Results save folder')
    ap.add_argument('--plot_show', default=True, type=bool)

    # === First stage: Parameter Estimation ===
    total_epochs = 10000
    decay_every = 2500
    milestones1 = [i for i in range(decay_every, total_epochs, decay_every)]

    ap.add_argument('--first_lr_disc', default=0.0001, type=float, help='Discriminator learning rate')
    ap.add_argument('--first_lr_para', default=0.001, type=float, help='Parameter learning rate')
    ap.add_argument('--first_n_dis', default=5, type=int, help='Update rate for discriminator')
    ap.add_argument('--first_batch_size', default=4, type=int, help='Batch size')
    ap.add_argument('--first_max_iter', default=total_epochs, type=int, help='Max iterations')
    ap.add_argument("--first_milestone", type=int, default=milestones1, help="LR decay milestones")

    # === Second stage: Noise Estimation ===
    total_epochs = 20000
    decay_every = 2500  # Faster decay for clearer (less noisy) environments
    # decay_every = 5000  # Slower decay for more turbid/noisy environments
    milestones2 = [i for i in range(decay_every, total_epochs, decay_every)]

    ap.add_argument('--second_lr_disc', default=0.00001, type=float, help='Discriminator learning rate')
    ap.add_argument('--second_lr_gen', default=0.00001, type=float, help='Generator learning rate')
    ap.add_argument('--second_n_dis', default=10, type=int, help='Update rate for discriminator')
    ap.add_argument('--second_batch_size', default=4, type=int, help='Batch size')
    ap.add_argument('--second_alpha', default=0.1, type=float, help='Loss weight lambda')
    ap.add_argument('--second_max_iter', default=total_epochs, type=int, help='Max iterations')
    ap.add_argument("--second_milestone", type=int, default=milestones2, help="LR decay milestones")

    # === Misc ===
    ap.add_argument('--seed', default=616, type=int, help='Random seed')
    ap.add_argument('--device', default='cuda', type=str, help='CUDA device')
    args = ap.parse_args()

    # === Set seed and device ===
    set_seeds(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Training Device:", args.device)

    # === Create folders ===
    first_save_path, second_save_path = create_folders(args.save_path)

    # ============================
    # First Stage Training (Parameter Estimation)
    # ============================

    print("[INFO] Starting First Stage: Parameter Estimation")

    train_dataset = SimpleDataset(args, scale=1)
    netB = BlendedModel().to(args.device)
    netD = DiscModel().to(args.device)

    # Separate sigma and other parameters (not necessary)
    sigma_params = [param for name, param in netB.named_parameters() if 'sigma_k' in name]
    other_params = [param for name, param in netB.named_parameters() if 'sigma_k' not in name]

    optimizerB = optim.Adam([
        {'params': sigma_params, 'lr': args.first_lr_para},
        {'params': other_params, 'lr': args.first_lr_para}
    ], weight_decay=0.0001, betas=(0.5, 0.999))

    optimizerD = optim.Adam(netD.parameters(), lr=args.first_lr_disc, weight_decay=0.0001, betas=(0.5, 0.999))

    schedulerB = optim.lr_scheduler.MultiStepLR(optimizerB, args.first_milestone, gamma=0.5)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, args.first_milestone, gamma=0.5)

    # Train
    first_gan_train(args, train_dataset, netB, netD, optimizerB, optimizerD, schedulerB, schedulerD)

    # ============================
    # Second Stage Training (Noise Estimation)
    # ============================

    print("[INFO] Starting Second Stage: Noise Estimation")

    train_dataset = SimpleDataset(args, scale=1)
    netB = BlendedModel().to(args.device)
    netD = DiscModel().to(args.device)
    netG = GenModel().to(args.device)
    netD_n = DiscModel_mul().to(args.device)

    # Load trained B model from first stage
    netB.load_state_dict(torch.load(os.path.join(first_save_path, 'B_simple.pt')))
    set_paras(netB, train_bool=False)

    optimizerG = optim.Adam(netG.parameters(), lr=args.second_lr_gen, weight_decay=0.0001, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.second_lr_disc, weight_decay=0.0001, betas=(0.5, 0.999))
    optimizerD_n = optim.Adam(netD_n.parameters(), lr=args.second_lr_disc, weight_decay=0.0001, betas=(0.5, 0.999))

    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, args.second_milestone, gamma=0.5)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, args.second_milestone, gamma=0.5)
    schedulerD_n = optim.lr_scheduler.MultiStepLR(optimizerD_n, args.second_milestone, gamma=0.5)

    # Train
    second_gan_train(args, train_dataset, netB, netG, netD, netD_n,
                     optimizerG, optimizerD, optimizerD_n,
                     schedulerG, schedulerD, schedulerD_n)

    print("[INFO] Training complete.")

