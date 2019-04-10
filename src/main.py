'''
@author: Aeolus
@url: x-fei.me
@time: 2019-04-09 17:06
'''
from src import config, tool, agent, logger

if __name__ == '__main__':
    config.init_path_config(__file__)
    pblog = logger.get_pblog(total=config.epoch)
    try:
        pblog.debug('###start###')
        if config.cmd == 'temp':
            pass
        elif config.cmd == 'train':
            args = {'desc':config.desc,
                    'cuda': config.cuda,
                    'num_workers': config.num_workers,
                    'dataset': config.dataset,
                    'img_size': config.img_size,
                    'ImageNet100_dir': config.ImageNet100_dir,
                    'CIFAR10_dir': config.CIFAR10_dir,
                    'model': config.model,
                    'optimizer': config.optimizer,
                    'action': config.action,
                    'model_dir': config.model_dir,
                    'best_model_dir': config.best_model_dir,
                    'tb_dir': config.tb_dir,
                    'pre_train': config.pre_train,
                    'epoch': config.epoch,
                    'batch_size': config.batch_size,
                    'lr': config.lr,
                    'lr_epoch': config.lr_epoch,
                    'no_eval': config.no_eval}
            agent.Trainer(args).train()
        else:
            raise ValueError('No cmd: {}'.format(config.cmd))
    except:
        pblog.exception('Exception Logged')
        exit(1)
    else:
        pblog.debug('###ok###')