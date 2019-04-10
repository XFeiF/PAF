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
        self.action.save_graph(self.model, self.img_size, self.tblog,
                               self.pblog)
        tool.check_mkdir(self.model_dir)
        tool.check_mkdir(self.best_model_dir)

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
            ll = len(self.action.eval_legend)
            c_right = np.zeros(ll, np.float32)
            c_sum = np.zeros(ll, np.float32)
            main_loss = 0
            for idx, (tx, ty) in enumerate(self.dataloader.train):
                tx, ty = tx.cuda(non_blocking=True), ty.cuda(non_blocking=True)
                # get network output logits
                logits = self.action.cal_logits(tx, self.model)
                # cal loss
                loss = self.action.cal_loss(ty, logits)
                # cal acc
                right_e, sum_e = self.action.cal_eval(ty, logits)
                # backward
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

                c_right += right_e
                c_sum += sum_e
                loss_l.append([ii.item() for ii in loss])
                loss_n.append(ty.size(0))
                main_loss += loss[0].item()
                self.pblog.pb(idx, dl_len, 'Loss: %.5f | Acc: %.3f%%' % (
                              main_loss / (idx + 1), c_right/c_sum))
            loss_l = np.array(loss_l).T
            loss_n = np.array(loss_n)
            loss = (loss_l * loss_n).sum(axis=1) / loss_n.sum()
            c_res = c_right / c_sum

            msg = 'Epoch: {:>3}'.format(epoch)
            loss_scalars = self.action.cal_scalars(loss,
                                                   self.action.loss_legend, msg,
                                                   self.pblog)
            self.tblog.add_scalars('loss', loss_scalars, epoch)

            msg = 'train->   '
            acc_scalars = self.action.cal_scalars(c_res,
                                                  self.action.eval_legend, msg,
                                                  self.pblog)
            self.tblog.add_scalars('eval/train', acc_scalars, epoch)

            if not self.no_eval:
                with torch.no_grad():
                    self.eval(epoch)

        path = os.path.join(self.model_dir, self.model_desc)
        self.action.save_model(self.ism, self.model, path, self.eval_best,
                               self.eval_best_epoch)
        self.pblog.debug('Training completed, save the last epoch model')
        temp = 'Result, Best: {:.2f}%, Epoch: {}'.format(self.eval_best,
                                                         self.eval_best_epoch)
        self.tblog.add_text('best', temp, self.epoch)
        self.pblog.info(temp)

    def eval(self, epoch):
        self.model.eval()
        ll = len(self.action.eval_legend)
        c_right = np.zeros(ll, np.float32)
        c_sum = np.zeros(ll, np.float32)
        dl_len = len(self.dataloader.eval)
        labels = []
        predictions = []
        for idx, (x, y) in enumerate(self.dataloader.eval):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = self.action.cal_logits(x, self.model)
            right_e, sum_e = self.action.cal_eval(y, logits)
            c_right += right_e
            c_sum += sum_e
            labels.extend(y.cpu().data)
            predictions.extend(logits.argmax(1).cpu().data)
            self.pblog.pb(idx, dl_len, 'Acc: %.3f %%' % (c_right / c_sum))
        msg = 'eval->    '
        c_res = c_right / c_sum
        acc_scalars = self.action.cal_scalars(c_res, self.action.eval_legend,
                                              msg, self.pblog)
        self.tblog.add_scalars('eval/eval', acc_scalars, epoch)
        cm_figure = self.action.log_confusion_matrix(labels, predictions,
                                                     self.dataloader.class_names)
        self.tblog.add_figure('Confusion Matrix', cm_figure, epoch)

        if c_res[0] > self.eval_best and epoch > 30:
            self.eval_best_epoch = epoch
            self.eval_best = c_res[0]
            path = os.path.join(self.best_model_dir, 'Best_' + self.model_desc)
            self.action.save_model(self.ism, self.model, path, self.eval_best,
                                   self.eval_best_epoch)
            self.pblog('Update the best model')
