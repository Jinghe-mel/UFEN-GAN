import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2


def first_gan_train(opt, datasets, netB, netD, optimizerB, optimizerD, lr_schedulerB, lr_schedulerD):
    save_path = os.path.join(opt.save_path, 'first')
    img_path = os.path.join(opt.save_path, 'first_img')
    img_list = []
    device = opt.device
    errB_list = []
    errD_list = []
    errD = 0
    iteration = 0
    data_loader = DataLoader(datasets, batch_size=opt.first_batch_size, shuffle=True, drop_last=True,
                             collate_fn=datasets.collate_fn)
    data_iterator = iter(data_loader)

    while iteration < opt.first_max_iter + 1:
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data = next(data_iterator)

        img, clear, dep = data
        img = img.to(device)
        clear = clear.to(device)
        dep = dep.to(device)
        if iteration % opt.first_n_dis == 0:
            netD.train()
            netD.zero_grad()

            real_image = img
            fake_make, _, _, _ = netB(clear, dep, torch.zeros_like(dep).to(dep.device))
            fake_make = torch.clip(fake_make, 0, 1)
            errD = netD.calc_dis_loss(fake_make, real_image)

            errD.backward()
            optimizerD.step()

        netB.train()
        netB.zero_grad()
        fake_make, _, _, back = netB(clear, dep, torch.zeros_like(dep).to(dep.device))
        errB = netD.calc_gen_loss(fake_make)
        errB.backward()
        optimizerB.step()
        iteration = iteration + 1
        if iteration % 1000 == 0:
            errD_list.append(errD)
            errB_list.append(errB)
            if opt.plot_show:
                img_list = [clear[0], fake_make[0], real_image[0]]
                visualize(img_list, img_path + '/' + str(iteration) + 'Combined Final.png')
            template = '[Iter:{:>2d}/{:<2d}] , errD={:5.2e}, errB={:5.2e}'
            print(template.format(iteration, opt.first_max_iter, errD, errB))
            print_parameters(netB)

            save_path_model = os.path.join(save_path, str(iteration) + 'B_simple' + '.pt')
            torch.save(netB.state_dict(), save_path_model)
            save_path_model = os.path.join(save_path, 'B_simple' + '.pt')
            torch.save(netB.state_dict(), save_path_model)

        lr_schedulerD.step()
        lr_schedulerB.step()


def visualize(image_list, name):
    def process_tensor(tensor):
        numpy_image = tensor.detach().cpu().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = (numpy_image * 255).astype('uint8')
        return numpy_image

    processed_images = [process_tensor(img) for img in image_list]
    combined_image = np.concatenate(processed_images, axis=1)
    cv2.imwrite(name, combined_image)


def print_parameters(model, print_grad=False):
    names = []
    values = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
            if name != "sigma_k":
                values.append(torch.sigmoid(param.data).item())
            else:
                values.append(torch.relu(param.data).item())
    print(", ".join(names))
    print(", ".join(f"{value:.4f}" for value in values))
