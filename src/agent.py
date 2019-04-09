'''
@author: Aeolus
@url: x-fei.me
@time: 2019-04-09 11:53
'''
import os
from os.path import join, exists
import shutil
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from src import tool, logger
from src import dataset as DataSet
from src import net as Model
from src import action as Action


# pblog = logger.ProgressBarLog()
class Trainer:

    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
        self.model_dir = args['model_dir']
        self.best_model_dir = args['best_model_dir']
        self.dataloader = DataSet.get_dataloader(args)
        self.no_eval = args['no_eval']
        self.img_size = args['img_size']
        args['mean'] = self.dataloader.mean
        args['std'] = self.dataloader.std
        args['num_classes'] = self.dataloader.num_classes

        self.action = Action.get_action(args)
        self.model = Model.get_net(args)

        self.model_desc = '{}_{}_{}_{}'. \
            format(args['dataset'], args['model'], args['action'], args['desc'])
        self.model_pkl = self.model_desc + '.ckpt'

        if args['pre_train']:
            state_dir = join(self.model_dir, self.model_desc)
            state = torch.load(state_dir, map_location='cpu')
            self.model.load_state_dict(state['net'])
        self.model.cuda()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.ism = True
        else:
            self.ism = False

        self.opt_type = args['optimizer']
        self.lr = args['lr']
        self.lr_epoch = args['lr_epoch']
        self.epoch = args['epoch']
        self.eval_best = 0
        self.eval_best_epoch = 0

        # logger
        self.pblog = logger.get_pblog()
        self.pblog.total = self.epoch
        self.tblog = SummaryWriter(join(args['tb_dir'], self.model_desc))
        self.save_graph()

    def __del__(self):
        if hasattr(self, 'tb_log'):
            self.tblog.close()

    def train(self):
        self.pblog.info(self.model_desc)
        optimizer = None
        for epoch in range(self.epoch):
            # get optimizer
            temp = self.action.update_opt(epoch, self.model, self.opt_type,
                                          self.lr, self.lr_epoch)
            if temp is not None:
                optimizer = temp

            self.model.train()
            loss_l = []
            loss_n = []
            dl_len = len(self.dataloader.train)
            main_loss = 0
            for idx, (tx, ty) in enumerate(self.dataloader.train):
                tx, ty = tx.cuda(non_blocking=True), ty.cuda(non_blocking=True)

                loss = self.action.cal_loss(tx, ty, self.model)
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

                loss_l.append([ii.item() for ii in loss])
                loss_n.append(ty.size(0))
                main_loss += loss[0].item()
                self.pblog.pb(idx, dl_len,
                              'Loss: %.5f' % (main_loss / (idx + 1)))
            loss_l = np.array(loss_l).T
            loss_n = np.array(loss_n)
            loss = (loss_l * loss_n).sum(axis=1) / loss_n.sum()
            msg = 'Epoch: {:>3}'.format(epoch)

            temp = dict()
            for n, s in zip(loss, self.action.loss_legend):
                msg += s.format(n)
                temp[s.split(':')[0][2:]] = n
            self.tblog.add_scalars('loss', temp, epoch)
            self.pblog.info(msg)
            if not self.no_eval:
                with torch.no_grad():
                    self.eval(epoch)
        tool.check_mkdir(self.model_dir)
        path = os.path.join(self.model_dir, self.model_desc)
        self.save_model(path)
        self.pblog.debug('training completed, save model')
        temp = 'Result, Best: {:.2f}%, Epoch: {}'.format(self.eval_best,
                                                         self.eval_best_epoch)
        self.tblog.add_text('best', temp, self.epoch)
        self.pblog.info(temp)

    def eval(self, epoch):
        self.model.eval()
        ll = len(self.action.eval_legend)
        if self.action.eval_on_train:
            c_right = np.zeros(ll, np.float32)
            c_sum = np.zeros(ll, np.float32)
            dl_len = len(self.dataloader.train)
            for idx, (x, y) in enumerate(self.dataloader.train):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                a, b = self.action.cal_eval(x, y, self.model)
                c_right += a
                c_sum += b
                self.pblog.pb(idx, dl_len, 'Acc: %.3f %%' % (c_right / c_sum))
            msg = 'train->   '
            tbd = dict()
            for n, s in zip(c_right / c_sum, self.action.eval_legend):
                msg += s.format(n)
                tbd[s.split(':')[0][2:]] = n
            self.tblog.add_scalars('eval/train', tbd, epoch)
            self.pblog.info(msg)
        c_right = np.zeros(ll, np.float32)
        c_sum = np.zeros(ll, np.float32)
        dl_len = len(self.dataloader.eval)
        for idx, (x, y) in enumerate(self.dataloader.eval):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            a, b = self.action.cal_eval(x, y, self.model)
            c_right += a
            c_sum += b
            self.pblog.pb(idx, dl_len, 'Acc: %.3f %%' % (c_right / c_sum))
        msg = 'eval->    '
        c_res = c_right / c_sum
        tbd = dict()
        for n, s in zip(c_res, self.action.eval_legend):
            msg += s.format(n)
            tbd[s.split(':')[0][2:]] = n
        self.tblog.add_scalars('eval/eval', tbd, epoch)
        self.pblog.info(msg)
        if c_res[0] > self.eval_best and epoch > 30:
            self.eval_best_epoch = epoch
            self.eval_best = c_res[0]
            if not exists(self.best_model_dir):
                os.makedirs(self.best_model_dir)
            path = os.path.join(self.best_model_dir, 'Best_' + self.model_desc)
            self.save_model(path)
            self.pblog.debug('update the best model')

    def save_model(self, path):
        if self.ism:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        state = {
            'net': state_dict,
            'acc': self.eval_best,
            'epoch': self.eval_best_epoch}
        torch.save(state, path)
        self.pblog.debug('Model saved')

    def save_graph(self):
        dummyInput = torch.randn([1, 3, self.img_size, self.img_size])
        self.tblog.add_graph(self.model, dummyInput)
        self.pblog.debug('Graph saved')
