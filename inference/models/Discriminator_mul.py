import torch.nn as nn
import torch
import munch
import torch.nn.functional as F
from .backbones import comomunit_new as networks
from .backbones.functions import init_net


def get_options():
    mo = munch.Munch()

    mo.num_scales = 2 # image scales
    mo.output_nc = 4
    mo.disc_dim = 64
    mo.disc_norm = 'none'
    mo.disc_activ = 'lrelu'
    mo.disc_n_layer = 4
    mo.disc_pad_type = 'reflect'
    mo.gan_mode = 'lsgan'
    mo.gpu_ids = 0
    # Initialization
    mo.init_type_gen = 'kaiming'
    mo.init_type_disc = 'normal'
    mo.init_gain = 0.02
    return mo


class DiscModel_mul(nn.Module):
    def __init__(self, scale=3, dim=4):
        super(DiscModel_mul, self).__init__()
        opt = get_options()
        opt.num_scales = scale
        opt.output_nc = dim
        self.opts = opt
        netD_A = networks.define_D_munit(opt.output_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ,
                                              opt.disc_n_layer,
                                              opt.gan_mode, opt.num_scales, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, opt.gpu_ids)
        self.netD_A = init_net(netD_A, init_type=opt.init_type_gen, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.opts.gan_mode == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.opts.gan_mode == 'nsgan':
                all0 = torch.zeros_like(out0)
                all1 = torch.ones_like(out1)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_mode)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.opts.gan_mode== 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.opts.gan_mode== 'nsgan':
                all1 = torch.ones_like(out0.data)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.opts.gan_mode)
        return loss

    def forward(self, x):
        out = self.netD_A.forward(x)
        return out


