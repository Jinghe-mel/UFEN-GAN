import torch.nn as nn
import munch

from .backbones import comomunit_new as networks


def get_options():
    mo = munch.Munch()

    mo.input_nc = 1
    mo.output_nc = 1
    mo.gen_dim = 64
    mo.style_dim = 8
    mo.gen_activ = 'relu'
    mo.n_downsample = 2
    mo.n_res = 4
    mo.gen_pad_type = 'reflect'
    mo.mlp_dim = 256
    mo.gan_mode = 'lsgan'
    mo.gpu_ids = 0
    # Initialization
    mo.init_type_gen = 'kaiming'
    mo.init_type_disc = 'normal'
    mo.init_gain = 0.02
    return mo


class GenModel(nn.Module):
    def __init__(self):
        super(GenModel, self).__init__()
        opt = get_options()
        self.opts = opt

        self.gen = networks.define_G_munit(opt.input_nc, opt.output_nc, opt.gen_dim, opt.style_dim, opt.n_downsample,
                                              opt.n_res, opt.gen_pad_type, opt.mlp_dim, opt.gen_activ, opt.init_type_gen,
                                              opt.init_gain, opt.gpu_ids)

    def forward(self, x):
        out = self.gen.forward(x)
        return out