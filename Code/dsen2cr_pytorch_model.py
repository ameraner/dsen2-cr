import os

import torch
import torch.nn as nn
from torch.nn import init

from .base_model import BaseModel

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"  # match IDs of nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"           # only set device 0 visible

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): torch.cuda.set_device(0)


class StackedResnetModel(BaseModel):

    def name(self):
        return 'stacked_resnet'

    def initialize(self, opt):
        # define network, initialize ("kaiming-uniform" = He init.)
        BaseModel.initialize(self, opt)
        self.netResNet = init_net(ResnetStackedArchitecture(), "kaiming-uniform", self.gpu_ids)

        self.model_names = ['ResNet']

        # define optimizer, see section 6.1.2
        # no learning rate schedule is applied, see section 7.1
        # torch has no Nesterov-Adam, so long take Adam with lr=1*10**-3, see section 6.1.3
        # --> saw 1*10**-3 being too large, changed it to 7*10**-4
        self.optimizer  = torch.optim.Adam(self.netResNet.parameters(), lr=7*10**-4, betas=(0.9, 0.999))
        self.optimizers = [self.optimizer]

        self.lambda_reg   = 1.0  # see section 6.6

        # initialize metrics calculated/updated on val split
        self.loss_eval_precision = 0
        self.loss_eval_recall    = 0
        self.loss_eval_f1        = 0

        self.loss_names   = ["cloud_adaptive", "target_reg", "CARL", "MAE", "RMSE", "eval_precision", "eval_recall", "eval_f1"]
        self.visual_names = ['sar_RGB', 'cloudy_RGB', 'cloud_mask_RGB', 'cloud_free_RGB', 'declouded_RGB']

    def set_input(self, input):
        # concatenate VV + VH SAR with 13-channel MS
        self.sar        = input["A"].to(self.device)
        self.cloudy     = input["B"].to(self.device)
        self.cloud_free = input["C"].to(self.device)
        # for synthetic data, the cloud mask will be the ground-truth synthetic cloud noise with values in [0, 1]
        self.cloud_mask = input["meta"]["cloud_mask"].to(self.device)
        self.input      = torch.cat([self.cloudy, self.sar], dim=1)

        # set images for visdom plotting
        self.sar_RGB        = self.sar[:, [0], ...] - 1  # map values from [0, 2] to [-1, 1]
        self.cloudy_RGB     = (self.cloudy[:, [3,2,1], ...] / 5) * 2 - 1  # map values from [0, 5] to [-1, 1]
        self.cloud_free_RGB = (self.cloud_free[:, [3,2,1], ...] / 5) * 2 - 1  # map values from [0, 5] to [-1, 1]
        self.cloud_mask_RGB = self.cloud_mask[:, [0], ...] * 2 - 1  # map values from [0, 1] to [-1, 1]

    def forward(self):
        self.declouded = self.netResNet(self.input)
        self.declouded_RGB = torch.clamp(self.declouded[:, [3, 2, 1], ...] / 5, 0, 1) * 2 - 1  # map values from [0, 5] to [-1, 1]

    def optimize_parameters(self):
        # forward
        self.forward()
        self.set_requires_grad([self.netResNet], True)
        self.optimizer.zero_grad()

        # define losses, see section 4.3
        N_tot = self.cloud_free.shape.numel()  # total number of pixels in batch images = B x C x H x W, see section 4.3.1

        self.loss_cloud_adaptive = torch.norm(self.cloud_mask     * (self.declouded - self.cloud_free) +        # cloudy area
                                              (1-self.cloud_mask) * (self.declouded - self.cloudy), 1) / N_tot  # cloud-free area
        self.loss_target_reg = torch.norm(self.declouded - self.cloud_free, 1) / N_tot
        self.loss_CARL = self.loss_cloud_adaptive + self.lambda_reg * self.loss_target_reg
        # define additional losses, see section 3.3
        # same as self.loss_target_reg, except we norm pixel values to [0, 1]
        self.loss_MAE  = torch.norm(torch.clamp(self.declouded / 5, 0, 1) -
                                    torch.clamp(self.cloud_free / 5, 0, 1), 1) / N_tot
        self.loss_RMSE = torch.sqrt(torch.norm(torch.clamp(self.declouded / 5, 0, 1) -
                                               torch.clamp(self.cloud_free / 5, 0, 1), 2) / N_tot)
        # first probe gradient norm before clipping
        """
        total_norm = 0
        for p in self.netResNet.parameters(): 
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(total_norm)
        """
        # clip gradients to protect against a batch of crazy samples
        torch.nn.utils.clip_grad_norm_(self.netResNet.parameters(), 5.)
        self.loss_CARL.backward()   # backprop loss
        self.optimizer.step()       # update parameters with gradients


def init_net(net, init_type="kaiming-uniform", gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


# see section 6.2.1
def init_weights(net, init_type="kaiming-uniform", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == "kaiming-uniform":
                init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class ResnetStackedArchitecture(nn.Module):

    def __init__(self):
        super(ResnetStackedArchitecture, self).__init__()
        self.F           = 256
        self.B           = 16
        self.kernel_size = 3
        self.padding_size= 1
        self.scale_res   = 0.1
        self.dropout     = False

        model = [nn.Conv2d(2+13, self.F, kernel_size=self.kernel_size, padding=self.padding_size, bias=True),
                 nn.ReLU(True)]
        # generate a given number of blocks
        for i in range(self.B):
            model += [ResnetBlock(self.F, use_dropout=self.dropout, use_bias=True,
                                  res_scale=self.scale_res, padding_size=self.padding_size)]

        model += [nn.Conv2d(self.F, 13, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # long-skip connection: add cloudy MS input (excluding the trailing two SAR channels) and model output
        return input[:, :-2, ...] + self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout, use_bias, res_scale=0.1, padding_size=1):
        super(ResnetBlock, self).__init__()
        self.res_scale = res_scale
        self.padding_size = padding_size
        self.conv_block = self.build_conv_block(dim, use_dropout, use_bias)

        # conv_block:
        #   CONV (pad, conv, norm),
        #   RELU (relu, dropout),
        #   CONV (pad, conv, norm)
    def build_conv_block(self, dim, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]
        conv_block += [nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # add residual mapping
        out = x + self.res_scale * self.conv_block(x)
        return out
