import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel
from torch.nn.parallel import DataParallel

class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)

        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()


    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        
        # 初始化时间和帧数计数器
        total_time = 0.0
        total_frames = 0
        
        # 预热：进行一次无效推理避免初始化时间影响结果
        with torch.no_grad():
            warmup_data = next(iter(self.test_loader))
            warmup_data = self.to_cuda(warmup_data)
            _ = self.net(warmup_data)
        
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            batch_size = data['img'].size(0)  # 获取当前batch的大小
            total_frames += batch_size
            
            # 同步CUDA操作并开始计时
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = self.net(data)
            
            # 同步并计算耗时
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            
            output = self.net.module.heads.get_lanes(output)
            predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        # 计算并记录FPS
        if total_time > 0:
            fps = total_frames / total_time
            self.recorder.logger.info(f'Inference FPS: {fps:.2f}')
        else:
            self.recorder.logger.warning('Total inference time is zero, FPS cannot be calculated.')
        
        metric = self.test_loader.dataset.evaluate(predictions,
                                                self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    # def test1(self):
    #     if not self.test_loader:
    #         self.test_loader = build_dataloader(self.cfg.dataset.test,
    #                                             self.cfg,
    #                                             is_train=False)
    #     self.net.eval()
    #     predictions = []
    #     for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
    #         data = self.to_cuda(data)
    #         with torch.no_grad():
    #             output = self.net(data)
    #             output = self.net.module.heads.get_lanes(output)
    #             predictions.extend(output)
    #         if self.cfg.view:
    #             self.test_loader.dataset.view(output, data['meta'])

    #     metric = self.test_loader.dataset.evaluate(predictions,
    #                                                self.cfg.work_dir)
    #     if metric is not None:
    #         self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def inference(self):
            # 构建测试数据加载器
            self.recorder.logger.info('Build test loader...')
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)

            # 将模型设置为评估模式
            self.net.eval()

            end = time.time()
            max_iter = len(self.test_loader)

            with torch.no_grad():
                for i, data in enumerate(self.test_loader):
                    date_time = time.time() - end
                    data = self.to_cuda(data)

                    # 进行推理
                    output = self.net(data)

                    batch_time = time.time() - end
                    end = time.time()

                    # 可以根据需要处理推理结果，这里简单记录推理时间
                    self.recorder.batch_time.update(batch_time)
                    self.recorder.data_time.update(date_time)

                    if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                        self.recorder.record('inference')

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
