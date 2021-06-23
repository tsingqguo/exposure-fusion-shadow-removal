import torch
from collections import OrderedDict
import time
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from .distangle_model import DistangleModel
from PIL import ImageOps,Image
import cv2


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return np.clip(image_numpy, 0, 255).astype(imtype)


class L_TV(nn.Module):
    def __init__(self):
        super(L_TV, self).__init__()
    def forward(self, x):
        _, _, h, w = x.size()
        count_h = (h - 1) * w
        count_w = (w - 1) * h

        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w - 1], 2).sum()
        return (h_tv / count_h + w_tv / count_w) / 2.0


class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target):
        _, cin, _, _ = pred.shape
        _, cout, _, _ = target.shape
        assert cin == 3 and cout == 3
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)
        kx = kx.repeat((3, 1, 1, 1))
        ky = ky.repeat((3, 1, 1, 1))

        pred_grad_x = F.conv2d(pred, kx, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, ky, padding=1, groups=3)
        target_grad_x = F.conv2d(target, kx, padding=1, groups=3)
        target_grad_y = F.conv2d(target, ky, padding=1, groups=3)

        loss = (
            nn.L1Loss(reduction=self.reduction)(
                pred_grad_x, target_grad_x) +
            nn.L1Loss(reduction=self.reduction)(
                pred_grad_y, target_grad_y))
        return loss * self.loss_weight


class PoissonGradientLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """L_{grad} = \frac{1}{2hw}\sum_{m=1}^{H}\sum_{n=1}{W}(\partial f(I_{Blend}) - 
                       (\partial f(I_{Source}) + \partial f(I_{Target})))_{mn}^2

           See **Deep Image Blending** for detail.
        """
        super(PoissonGradientLoss, self).__init__()
        self.reduction = reduction

    def forward(self, source, target, blend, mask):
        f = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3).to(target)
        f = f.repeat((3, 1, 1, 1))
        grad_s = F.conv2d(source, f, padding=1, groups=3) * mask
        grad_t = F.conv2d(target, f, padding=1, groups=3) * (1 - mask)
        grad_b = F.conv2d(blend, f, padding=1, groups=3)
        return nn.MSELoss(reduction=self.reduction)(grad_b, (grad_t + grad_s))


class FusionModel(DistangleModel):
    def name(self):
        return 'fusion net cvpr 21'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='expo_param')
        parser.add_argument('--wdataroot',default='None',  help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--use_our_mask', action='store_true')
        parser.add_argument('--mask_train',type=str,default=None)
        parser.add_argument('--mask_test',type=str,default=None)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G_param', 'alpha', 'rescontruction']
        self.visual_names = ['input_img', 'litgt', 'alpha_pred', 'out', 'final', 'outgt']
        self.model_names = ['G', 'M']
        opt.output_nc = 3 

        self.ks = ks = opt.ks
        self.n = n = opt.n
        self.shadow_loss = opt.shadow_loss
        self.tv_loss = opt.tv_loss
        self.grad_loss = opt.grad_loss
        self.pgrad_loss = opt.pgrad_loss

        self.netG = networks.define_G(4, 2 * 3, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netM = networks.define_G(1 + 3 + n * 3, ((1 + n) * 3) * 3 * ks * ks, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG.to(self.device)
        self.netM.to(self.device)
        print(self.netG)
        print(self.netM)
        if self.isTrain:
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []

            if opt.optimizer == 'adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
                self.optimizer_M = torch.optim.Adam(self.netM.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            elif opt.optimizer == 'sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), momentum=0.9,
                                                   lr=opt.lr, weight_decay=1e-5)
                self.optimizer_M = torch.optim.SGD(self.netM.parameters(), momentum=0.9,
                                                   lr=opt.lr, weight_decay=1e-5) 
            else:
                assert False

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_M)
   
    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.imname = input['imname']
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.shadow_mask = (self.shadow_mask > 0.9).type(torch.float) # * 2 - 1
        self.nim = self.input_img.shape[1]
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_mask_3d = (self.shadow_mask > 0).type(torch.float).expand(self.input_img.shape)
        self.shadow_mask_dilate = input['B_dilate'].to(self.device)
        self.shadow_mask_erode = input['B_erode'].to(self.device)

    def forward(self):
        inputG = torch.cat([self.input_img, self.shadow_mask], 1)
        shadow_param_pred = self.netG(inputG)

        n = shadow_param_pred.shape[0]
        w = inputG.shape[2]
        h = inputG.shape[3]

        addgt = self.shadow_param[:, [0, 2, 4]]
        mulgt = self.shadow_param[:, [1, 3, 5]]
        addgt = addgt.view(n, 3, 1, 1).expand((n, 3, w, h))
        mulgt = mulgt.view(n, 3, 1, 1).expand((n, 3, w, h))

        base_shadow_param_pred = shadow_param_pred[:, :2 * 3] # shadow_param_pred.view((n, self.n * 3, 2, 1, 1))
        self.base_shadow_param_pred = base_shadow_param_pred

        shadow_image = self.input_img.clone() / 2 + 0.5
        base_shadow_output = shadow_image * base_shadow_param_pred[:, :3].view((n, 3, 1, 1)) + \
                             base_shadow_param_pred[:, 3:].view((n, 3, 1, 1))
        shadow_output_list = []
        for i in range(0, self.n - 1):
            if i % 2 == 0:
                scale = 1 + i * 0.01
            else:
                scale = 1 - i * 0.01
            shadow_output_list.append(base_shadow_output * scale)
        shadow_output = torch.cat([base_shadow_output] + shadow_output_list, dim=1)
        self.lit = torch.cat([base_shadow_output] + shadow_output_list, dim=-1) * 2 - 1

        shadow_output = shadow_output * 2 - 1
        self.shadow_output = shadow_output

        self.litgt = self.input_img.clone() / 2 + 0.5
        self.litgt = (self.litgt * mulgt + addgt) * 2 - 1 # [-1, 1]

        inputM = torch.cat([self.input_img, shadow_output, self.shadow_mask], 1)
        out = torch.cat([self.input_img, shadow_output], 1)
        out = out / 2 + 0.5
        out_matrix = F.unfold(out, stride=1, padding=self.ks // 2, kernel_size=self.ks) # N, C x \mul_(kernel_size), L

        kernel = self.netM(inputM) # b, (3+1)*n * 3 * ks * ks, Tanh

        b, c, h, w = self.input_img.shape
        output = []
        for i in range(b):
            # feature = out[i, ...]
            feature = out_matrix[i, ...] # ((1 + n) * 3) * ks * ks, L
            weight = kernel[i, ...] # ((1 + n) * 3) * 3 * ks * ks, H, W
            feature = feature.unsqueeze(0) # 1, C, L
            weight = weight.view((3, (self.n + 1) * 3 * self.ks * self.ks, h * w))
            weight = F.softmax(weight, dim=1)
            iout = feature * weight # (3, C, L)
            iout = torch.sum(iout, dim=1, keepdim=False)
            iout = iout.view((1, 3, h, w))

            output.append(iout)
        self.final = torch.cat(output, dim=0) * 2 -1


    def backward(self):
        criterion = self.criterionL1
        lambda_ = self.opt.lambda_L1

        addgt = self.shadow_param[:, [0, 2, 4]] # [b, 3]
        mulgt = self.shadow_param[:, [1, 3, 5]] # [b, 3]

        loss_G_param_mul = self.MSELoss(self.base_shadow_param_pred[:, :3], mulgt) * lambda_
        loss_G_param_add = self.MSELoss(self.base_shadow_param_pred[:, 3:], addgt) * lambda_
        self.loss_G_param = (loss_G_param_add + loss_G_param_mul) / 2.0 * self.shadow_loss

        if self.tv_loss > 0:
            tv_loss = L_TV()(self.final - self.shadowfree_img) * lambda_ * self.tv_loss
        else:
            tv_loss = 0.0
        
        if self.grad_loss > 0:
            grad_loss = GradientLoss()(self.final, self.shadowfree_img) * lambda_ * self.grad_loss
        else:
            grad_loss = 0.0

        if self.pgrad_loss > 0:
            pgrad_loss = PoissonGradientLoss()(target=self.input_img, blend=self.final,
                                               source=self.shadowfree_img, mask=self.shadow_mask_dilate) \
                                               * lambda_ * self.pgrad_loss
        else:
            pgrad_loss = 0.0

        self.loss_rescontruction = criterion(self.final, self.shadowfree_img) * lambda_
        self.loss = self.loss_rescontruction + self.loss_G_param + tv_loss + grad_loss + pgrad_loss
        self.loss.backward()

    def optimize_parameters(self):
        self.netM.zero_grad()
        self.netG.zero_grad()
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_M.zero_grad()
        self.backward()
        self.optimizer_G.step()
        self.optimizer_M.step()
    
    def zero_grad(self):
        self.netM.zero_grad()
        self.netG.zero_grad()
        self.optimizer_G.zero_grad()
        self.optimizer_M.zero_grad()

    def vis(self, e, s, path='', eval=False):
        if len(path) > 0:
            save_dir = os.path.join(self.save_dir, path)
        else:
            save_dir = self.save_dir
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        shadow = self.input_img
        output = self.final
        gt = self.shadowfree_img
        if eval:
            img = self.final[0, ...]
            filename = os.path.join(save_dir, self.imname[0])
        else:
            img = torch.cat([shadow, output, gt, self.litgt, self.lit], axis=-1)[0, ...]
            filename = os.path.join(save_dir, "epoch_%d_step_%d.png" % (e, s))
        img = tensor2im(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)

