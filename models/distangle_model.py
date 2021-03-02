import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util



class DistangleModel(BaseModel):
    def name(self):
        return 'DistangleModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='expo_param')
        parser.set_defaults(netG='RESNEXT')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['input_img', 'shadow_mask','out','outgt']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        opt.output_nc= 3 if opt.task=='sr' else 1 #3 for shadow removal, 1 for detection
        self.netG = networks.define_G(4, opt.output_nc, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG.to(self.device)
        print(self.netG)
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-5)
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   

    def forward(self):
        inputG = torch.cat([self.input_img, self.shadow_mask], 1)
        self.Gout = self.netG(inputG)
        self.lit = self.input_img.clone() / 2 + 0.5
        n = self.Gout.shape[0]

        add = add.view(n, 3, 1, 1).expand((n, 3, 256, 256))
        mul = mul.view(n, 3, 1, 1).expand((n, 3, 256, 256))

        self.litgt = (self.input_img.clone() + 1) / 2
        self.lit = self.lit * mul + add
        self.litgt = self.litgt * mulgt + addgt
        self.out = (self.input_img / 2 + 0.5) * (1 - self.shadow_mask_3d) + self.lit * self.shadow_mask_3d
        self.out = self.out * 2 - 1
        self.outgt = (self.input_img / 2 + 0.5) * (1 - self.shadow_mask_3d) + self.litgt * self.shadow_mask_3d
        self.outgt = self.outgt * 2 - 1
        self.alpha = torch.mean(self.shadowfree_img / self.lit,dim=1,keepdim=True)


    def get_prediction(self,input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        self.Gout = self.netG(inputG)
        self.lit = self.input_img.clone()/2+0.5
        add = self.Gout[:,[0,2,4]]
        mul = self.Gout[:,[1,3,5]]
        n = self.Gout.shape[0]
        add = add.view(n,3,1,1).expand((n,3,256,256))
        mul = mul.view(n,3,1,1).expand((n,3,256,256))
        self.lit = self.lit*mul + add
        self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
        self.out = self.out*2-1
        return util.tensor2im(self.out,scale =0) 

    def backward_G(self):
        criterion = self.criterionL1 if self.opt.task =='sr' else self.bce
        lambda_ = self.opt.lambda_L1 if self.opt.task =='sr' else 1
        self.loss_G = criterion(self.Gout, self.shadow_param) * lambda_
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
