import torch
import math
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torch.autograd import Variable


def second_gan_train(opt, datasets, netB, netG, netD, netD_n, optimizer, optimizerD, optimizerD_n,
                     lr_scheduler, lr_schedulerD, lr_schedulerD_n):
    save_path = os.path.join(opt.save_path, 'second')
    img_path = os.path.join(opt.save_path, 'second_img')

    device = opt.device
    grid_size = 80  # Patch size for local discriminator

    data_loader = DataLoader(datasets, batch_size=opt.first_batch_size, shuffle=True, drop_last=True,
                             collate_fn=datasets.collate_fn)
    data_iterator = iter(data_loader)
    iteration = 0

    while iteration < opt.second_max_iter + 1:
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data = next(data_iterator)

        img, clear, dep = data
        img = img.to(device)
        clear = clear.to(device)
        dep = dep.to(device)

        real_image = img
        random_noise = Variable(torch.randn(clear.size(0), 1, 10).to(device))
        input_make = random_noise

        if iteration % opt.second_n_dis == 0:
            netD.train()
            netD.zero_grad()

            fake_noise = netG(input_make.detach())
            fake_make, _, _, _ = netB(clear, dep, fake_noise)
            fake_image = fake_make

            fake_noise2 = netG(input_make.detach())
            N, B, H, W = fake_noise.shape

            # Sample patches for even distribution
            num_grids_h = H // grid_size
            num_grids_w = W // grid_size

            # Fixed patch (same location)
            grid_i = torch.randint(low=0, high=num_grids_h, size=(1,))
            grid_j = torch.randint(low=0, high=num_grids_w, size=(1,))
            start_i = grid_i * grid_size
            start_j = grid_j * grid_size
            fixed_patches = fake_noise2[:, :, start_i:start_i + grid_size,
                            start_j:start_j + grid_size].squeeze().view(N*B, grid_size, grid_size).unsqueeze(dim=0)

            # Random patches (different locations)
            random_patches = []
            for n in range(N):
                grid_i_n = torch.randint(low=0, high=num_grids_h, size=(1,))
                grid_j_n = torch.randint(low=0, high=num_grids_w, size=(1,))
                start_i_n = grid_i_n * grid_size
                start_j_n = grid_j_n * grid_size
                patch = fake_noise2[n, :, start_i_n:start_i_n + grid_size, start_j_n:start_j_n + grid_size]
                random_patches.append(patch)

            random_patches = torch.stack(random_patches)
            random_patches = random_patches.squeeze().view(N*B, grid_size, grid_size).unsqueeze(dim=0)
            errD_n = netD_n.calc_dis_loss(random_patches, fixed_patches) * opt.second_alpha
            errD = netD.calc_dis_loss(fake_image, real_image)

            errD.backward()
            errD_n.backward()
            optimizerD.step()
            optimizerD_n.step()

        fake_noise = netG(input_make.detach())
        fake_make, _, _, _ = netB(clear, dep, fake_noise)
        fake_image = fake_make

        fake_noise2 = netG(input_make.detach())
        N, _, H, W = fake_noise.shape
        num_grids_h = H // grid_size
        num_grids_w = W // grid_size

        random_patches = []
        for n in range(N):
            grid_i_n = torch.randint(low=0, high=num_grids_h, size=(1,))
            grid_j_n = torch.randint(low=0, high=num_grids_w, size=(1,))
            start_i_n = grid_i_n * grid_size
            start_j_n = grid_j_n * grid_size
            patch = fake_noise2[n, :, start_i_n:start_i_n + grid_size, start_j_n:start_j_n + grid_size]
            random_patches.append(patch)
        random_patches = torch.stack(random_patches)
        random_patches = random_patches.squeeze().view(N*B, grid_size, grid_size).unsqueeze(dim=0)

        netG.train()
        netG.zero_grad()
        errG_g = netD.calc_gen_loss(fake_image)
        errG_n = netD_n.calc_gen_loss(random_patches) * opt.second_alpha
        err = errG_g + errG_n

        err.backward()
        optimizer.step()

        iteration = iteration + 1

        if iteration % 1000 == 0:
            if opt.plot_show:
                fake_noise = torch.stack([fake_noise, fake_noise, fake_noise], dim=1).squeeze()
                vir_noise = (fake_noise - fake_noise.min())/(fake_noise.max()-fake_noise.min())
                img_list = [clear[0], vir_noise[0], fake_image[0], real_image[0]]
                visualize(img_list, img_path + '/' + str(iteration) + 'Combined Final.png')

            mean_a = fake_noise.mean()
            template = '[{:0>5d}/{:0>5d}, errG={:5.1e}, errD={:5.2e}, errGn={:5.2e}, ' \
                       'errDn={:5.2e}, mean={:5.2e}]'
            print(template.format(iteration, opt.second_max_iter, errG_g, errD, errG_n, errD_n, mean_a))

            save_path_model = os.path.join(save_path, str(iteration) + 'G_simple' + '.pt')
            torch.save(netG.state_dict(), save_path_model)
            save_path_model = os.path.join(save_path, 'G_simple' + '.pt')
            torch.save(netG.state_dict(), save_path_model)

        lr_scheduler.step()
        lr_schedulerD.step()
        lr_schedulerD_n.step()


def visualize(image_list, name):
    def process_tensor(tensor):
        numpy_image = tensor.detach().cpu().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = (numpy_image * 255).astype('uint8')
        return numpy_image

    processed_images = [process_tensor(img) for img in image_list]
    combined_image = np.concatenate(processed_images, axis=1)
    cv2.imwrite(name, combined_image)

