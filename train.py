import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import accelerate
from accelerate import DistributedDataParallelKwargs
import hydra
from hydra.utils import instantiate
import logging

# fix random seeds for reproducibility
logger = logging.getLogger(__name__)
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@hydra.main(version_base="1.2", config_path="cfg", config_name="train")
def main(config):
    accelerate.utils.set_seed(2023)
    logger = config.get_logger('train')

    # setup data_loader instances
    train_dataset = instantiate(config.dataset.data,mode='train',augmentation=True)
    valid_dataset = instantiate(config.dataset.data,mode='valid',augmentation=True)

    train_loader = instantiate(config.dataset.loader,dataset=train_dataset,shuffle=False)
    valid_loader = instantiate(config.dataset.loader,dataset=valid_dataset,shuffle=True) 

    # build model architecture, then print to console
    model = instantiate(config.model.network)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = instantiate(module_loss, config['loss'])
    metrics = [instantiate(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # setting  optimizer & scalduler
    opt = instantiate(config.model.optimizer, params=model.parameters())
    sch = instantiate(config.model.sch, optimizer=opt, steps_per_epoch=len(train_loader))

    trainer = Trainer(model, criterion, metrics, opt,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      valid_data_loader=valid_loader,
                      lr_scheduler=sch)

    trainer.fit()


if __name__ == '__main__':
    main()