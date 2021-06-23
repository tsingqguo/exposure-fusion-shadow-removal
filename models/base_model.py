import os
import time
import torch
from collections import OrderedDict
from . import networks
import util.util as util
import numpy as np
class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    

    def train(self):
        print('switching to training mode')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
    # make models eval mode during test time
    def eval(self):
        print('switching to testing mode')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.epoch=0
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        # if opt.resize_or_crop != 'scale_width':
        #     torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    #def set_input(self, input):
    #    self.input = input


    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        #self.shadow_mask = (self.shadow_mask==1).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        #self.shadow_mask_3d_over = (self.shadow_mask_over>0).type(torch.float).expand(self.input_img.shape)

    def get_prediction(self,input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        out = self.netG(inputG)
        return util.tensor2im(out)

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        print(self.name)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train or opt.finetuning:
            print("LOADING %s"%(self.name))
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)


    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self,loss=None):
        for scheduler in self.schedulers:
            if not loss:
                scheduler.step()
            else:
                scheduler.step(loss)

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        t= time.time()
        nim = self.shadow.shape[0]
        visual_ret = OrderedDict()
        all =[]
        for i in range(0,min(nim-1,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:])
                        row.append(im)
            row=tuple(row)
            all.append(np.hstack(row))
        all = tuple(all)
        
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])
    
    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                if hasattr(self, 'loss_' + name):
                    errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module,key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch, save_dir=None):
        print(epoch)
        if save_dir is None:
            save_dir = self.save_dir
        
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(save_dir, load_filename)
                if self.opt.finetuning:

                    load_filename = '%s_net_%s.pth' % (self.opt.finetuning_epoch, name)
                    load_path = os.path.join(self.opt.finetuning_dir, load_filename)

                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                #
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
