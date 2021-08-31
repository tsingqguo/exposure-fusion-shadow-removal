from collections import OrderedDict
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from PIL import Image
import time
import math
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from skimage.color import rgb2lab
import numpy as np
import torch
import torch.nn as nn
import os
import shutil

import logging

opt = TrainOptions().parse()

opt.phase = 'train/train_'
opt.serial_batches = False
train_data_loader = CreateDataLoader(opt)
train_dataset = train_data_loader.load_data()
train_dataset_size = len(train_data_loader)

opt.phase = 'test/test_'
opt.batch_size = 1
opt.serial_batches = True
test_data_loader = CreateDataLoader(opt)
test_dataset = test_data_loader.load_data()
test_dataset_size = len(test_data_loader)

model = create_model(opt)
model.setup(opt)
if opt.load_dir and opt.load_dir != 'None':
    print('load fusion net from:', opt.load_dir)
    model.load_networks('latest', opt.load_dir)

# Set logger
msg = []
logger = logging.getLogger('%s' % opt.name)
logger.setLevel(logging.INFO)
if not os.path.isdir(model.save_dir):
    msg.append('%s not exist, make it' % model.save_dir)
    os.mkdir(args.dir)
log_file_path = os.path.join(model.save_dir, 'log.log')
if os.path.isfile(log_file_path):
    target_path = log_file_path + '.%s' % time.strftime("%Y%m%d%H%M%S")
    msg.append('Log file exists, backup to %s' % target_path)
    shutil.move(log_file_path, target_path)
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


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
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return np.clip(image_numpy, 0, 255).astype(imtype)


def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab


mse_criterion = nn.MSELoss()
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    model.epoch = epoch

    model.train()
    for i, data in enumerate(train_dataset):
        iter_start_time = time.time()
        total_steps += 1
        model.set_input(data)
        # model.zero_grad()
        model.optimize_parameters()

        if total_steps % 10 == 0:
            # Do log
            train_loss = model.loss.detach().item()
            train_mse = mse_criterion(model.final, model.shadowfree_img).detach().item()
            logger.info('[Train] [Epoch] %d [Steps] %d | loss : %.3f' % (epoch, total_steps, train_loss))

    if (epoch and epoch % 30 == 0) or (epoch == opt.niter + opt.niter_decay):
        model.eval()
        eval_shadow_rmse = 0
        eval_nonshadow_rmse = 0
        eval_rmse = 0
        eval_loss = 0
        for i, data in enumerate(test_dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.forward()

            eval_loss += model.loss.detach().item()
            diff = calc_RMSE(tensor2im(model.shadowfree_img), tensor2im(model.final))
            mask = model.shadow_mask.data[0].cpu().float().numpy()[..., None][0, ...]

            if mask.sum() < 2:
                continue
            shadow_rmse = np.sqrt(1.0 * (np.power(diff, 2) * mask).sum(axis=(0, 1)) / mask.sum())
            nonshadow_rmse = np.sqrt(1.0 * (np.power(diff, 2) * (1 - mask)).sum(axis=(0, 1)) / (1 - mask).sum())
            whole_rmse = np.sqrt(np.power(diff, 2).mean(axis=(0, 1)))

            # (256, 256, 3) (3,) (3,) (256, 256, 1) (256, 256, 3)
            # print(diff.shape, whole_rmse.shape, shadow_rmse.shape, mask.shape, (diff * mask).shape)

            eval_shadow_rmse += shadow_rmse.sum()
            eval_nonshadow_rmse += nonshadow_rmse.sum()
            eval_rmse += whole_rmse.sum()

            model.zero_grad()

            if i % 20 == 0:
                model.vis(epoch, i)

        logger.info('[Eval] [Epoch] %d | loss : %.3f | rmse : %.3f | shadow_rmse : %.3f | nonshadow_rmse : %.3f' %
                    (epoch, eval_loss / len(test_dataset), eval_rmse / len(test_dataset),
                     eval_shadow_rmse / len(test_dataset), eval_nonshadow_rmse / len(test_dataset)))

    if epoch and epoch % 50 == 0 or (epoch == opt.niter + opt.niter_decay):
        logger.info('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    spt_time = time.time() - epoch_start_time
    lft_time = (opt.niter + opt.niter_decay - epoch) * spt_time
    logger.info('End of epoch %d / %d | Time Taken: %d sec | eta %.2f' %
                (epoch, opt.niter + opt.niter_decay, spt_time, lft_time / 3600.0))
    model.update_learning_rate()
