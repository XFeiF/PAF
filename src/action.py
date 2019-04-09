'''
@author: Aeolus
@url: x-fei.me
@time: 2019-04-09 11:40
'''
import numpy as np
import torch
import torch.nn.functional as F


class BaseAction:
    loss_legend = ['| loss: {:0<10.8f}']
    eval_on_train = True
    eval_legend = ['| acc: {:0<5.3f}%']

    @staticmethod
    def cal_loss(x, y, net):
        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y)
        return loss,

    @staticmethod
    def cal_eval(x, y, net):
        count_right = np.empty(1, np.float32)
        count_sum = np.empty(1, np.float32)
        y_hat = net(x).argmax(1)
        count_right[0] = (y_hat == y).sum().item()
        count_sum[0] = y.size(0)
        return 100 * count_right, count_sum

    @staticmethod
    def update_opt(epoch, net, opt_type, lr=1e-2, lr_epoch=35):
        decay = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        if epoch % lr_epoch == 0:
            times = int(epoch / lr_epoch)
            times = len(decay) - 1 if times >= len(decay) else times
            if opt_type == 'sgd':
                return torch.optim.SGD(net.parameters(), lr=lr * decay[times],
                                       momentum=0.9,
                                       weight_decay=5e-4)
            elif opt_type == 'adam':
                torch.optim.Adam(net.parameters(), lr=lr * decay[times])
        else:
            return None

def get_action(args):
    action = args['action']
    if action == 'base':
        return BaseAction()
    else:
        raise ValueError('No action: {}'.format(action))
