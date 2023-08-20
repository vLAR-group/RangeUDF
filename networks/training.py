from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time
import datetime
from .loss import compute_loss, IoUCalculator, kendall1, kendall2, kendall3, kendall4
import itertools
import torchsummary as summary


class Trainer(object):

    def __init__(self, model, opt, device, train_dataset, val_dataset, exp_name, log_dir=None, optimizer='Adam', lr = 1e-4, gamma=1, threshold = 0.1, checkpoint=None):
        self.checkpoint = checkpoint

        model = nn.DataParallel(model)
        
        self.model = model.cuda()
        # raise ValueError
        if opt.joint_mode == 'kendall1':
            self.kendall = kendall1()
            self.kendall = self.kendall.cuda()
            param_group = itertools.chain(self.model.parameters(), self.kendall.parameters())
            
        elif opt.joint_mode == 'kendall2':
            self.kendall = kendall2()
            self.kendall = self.kendall.cuda()
            param_group = itertools.chain(self.model.parameters(), self.kendall.parameters())
            
        elif opt.joint_mode == 'kendall3':
            self.kendall = kendall3()
            self.kendall = self.kendall.cuda()
            param_group = itertools.chain(self.model.parameters(), self.kendall.parameters())
            
        elif opt.joint_mode == 'kendall4':
            self.kendall = kendall4()
            self.kendall = self.kendall.cuda()
            param_group = itertools.chain(self.model.parameters(), self.kendall.parameters())

        elif opt.joint_mode == 'naive':
            self.kendall = None
            param_group = self.model.parameters()
        else:
            raise ValueError

        self.opt = opt
        if gamma == 1:
            self.gamma = 1
        elif gamma == 2:
            self.gamma = 0.1 ** (1 / 10000)
        elif gamma == 3:
            self.gamma = 0.95 ** (1 / 500)
        else:
            raise ValueError

        self.device = device
        
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(param_group, lr= lr)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.gamma)
            # self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opt.num_epochs, eta_min=lr/100, last_epoch=-1)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(param_group)
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(param_group, momentum=0.9)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if log_dir is None:
            start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            start_time = log_dir

        print('start time:', start_time)
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name) + start_time + '/'
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.max_dist = threshold


    def train_step(self,batch, steps, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, rec_loss, sem_loss, acc, on_acc, off_acc, on_rec, off_rec, on_sem, off_sem = compute_loss(self.model, batch, self.opt, 'train', kendall=self.kendall)
        loss.backward()
        self.optimizer.step()
        print("Epoch {} {}/{} Current loss: {}/{}, recon {}/{}, sem {}/{}, on_recon {}/{}, off_recon {}/{}, on_sem {}/{}, off_sem {}/{}, acc {}, on acc {}, off acc {}".format(epoch,
                                                                                                                 steps,
                                                                                                                 len(self.train_data_loader),  
                                                                                                                 loss.item() / (self.opt.num_sample_points_training + self.opt.num_points), 
                                                                                                                 loss.item(),
                                                                                                                 rec_loss / (self.opt.num_sample_points_training + self.opt.num_points), 
                                                                                                                 rec_loss, 
                                                                                                                 sem_loss / (self.opt.num_sample_points_training + self.opt.num_points), 
                                                                                                                 sem_loss, 
                                                                                                                 on_rec / self.opt.num_points, 
                                                                                                                 on_rec,
                                                                                                                 off_rec / self.opt.num_sample_points_training, 
                                                                                                                 off_rec,
                                                                                                                 on_sem / self.opt.num_points, 
                                                                                                                 on_sem,
                                                                                                                 off_sem / self.opt.num_sample_points_training, 
                                                                                                                 off_sem,
                                                                                                                 acc, 
                                                                                                                 on_acc, 
                                                                                                                 off_acc))
        return loss.item(), rec_loss.item(), sem_loss.item(), acc, on_acc, off_acc


    def train_model(self, epochs):
        self.train_data_loader = self.train_dataset.get_loader()
        start, training_time = self.load_checkpoint()
        iteration_start_time = time.time()

        for epoch in range(start, epochs):
            sum_rec_loss = 0
            sum_sem_loss = 0
            sum_acc = 0
            sum_on_acc = 0
            sum_off_acc = 0
            print('Start epoch {}'.format(epoch))
            steps = 0
            for batch in self.train_data_loader:

                # evaluation and save checkpoints
                iteration_duration = time.time() - iteration_start_time
                # if iteration_duration > 60 * 30:  # eve model every X min and at start

                # training
                steps += 1
                loss, rec_loss, sem_loss, acc, on_acc, off_acc = self.train_step(batch, steps, epoch)
                sum_sem_loss += sem_loss
                sum_rec_loss += rec_loss
                sum_acc += acc
                sum_on_acc += on_acc
                sum_off_acc += off_acc

            # end of epoch
            # val_loss, val_acc = self.compute_val_loss()
            self.writer.add_scalar('training acc batch avg', sum_acc / len(self.train_data_loader), epoch)
            self.writer.add_scalar('training on_acc batch avg', sum_on_acc / len(self.train_data_loader), epoch)
            self.writer.add_scalar('training off_acc batch avg', sum_off_acc / len(self.train_data_loader), epoch)
            self.writer.add_scalar('training rec_loss batch avg', sum_rec_loss / len(self.train_data_loader), epoch)
            self.writer.add_scalar('training sem_loss batch avg', sum_sem_loss / len(self.train_data_loader), epoch)
            print('training acc batch avg', sum_acc / len(self.train_data_loader))
            print('training on_acc batch avg', sum_on_acc / len(self.train_data_loader))
            print('training off_acc batch avg', sum_off_acc / len(self.train_data_loader))
            print('training rec_loss batch avg', sum_rec_loss / len(self.train_data_loader))
            print('training sem_loss batch avg', sum_sem_loss / len(self.train_data_loader))

            if epoch % 1 ==0:
                training_time += iteration_duration
                iteration_start_time = time.time()

                val_loss, val_acc, val_rec_loss, val_sem_loss, val_on_acc, val_off_acc = self.compute_val_loss()
               
                self.save_checkpoint(epoch, training_time,True)
                if 'syn' in self.opt.exp_name or 'rec' in self.opt.task:
                    mean_iou=0.0
                    overall_acc=0.0
                    mean_acc=0.0
                    iou_list=[0.0]
                    acc_list=[0.0]
                else:
                    mean_iou, overall_acc, mean_acc, iou_list, acc_list = self.iou_cal.compute_iou()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])
                    self.save_checkpoint(epoch, training_time,False)

                self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                self.writer.add_scalar('val acc batch avg', val_acc, epoch)
                self.writer.add_scalar('val iou batch avg', mean_iou, epoch)
                self.writer.add_scalar('val rec_loss batch avg', val_rec_loss, epoch)
                self.writer.add_scalar('val sem_loss batch avg', val_sem_loss, epoch)
                print('val loss batch avg: ', val_loss)
                print('val rec_loss batch avg: ', val_rec_loss, self.opt.num_sample_points_training, val_rec_loss / self.opt.num_sample_points_training)
                print('val sem_loss batch avg: ', val_sem_loss)
                print('val acc batch avg: ', val_acc)
                print('val on acc batch avg: ', val_on_acc)
                print('val off acc batch avg: ', val_off_acc)
                print('mean iou: ', mean_iou)
                print('mean acc: ', mean_acc)
                print('overall acc: ', overall_acc)
                print(iou_list)
                print(acc_list)
                print('End of validation {}'.format(epoch))
            # self.scheduler.step()


    def eval_model(self):
        import pickle
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0,0
        path = self.checkpoint_path + 'checkpoint_latest.tar'

        if self.checkpoint is not None:
            path = self.checkpoint_path + 'checkpoint_best.tar'
        print('Loaded checkpoint from: {}'.format(path))

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.kendall is not None:
            self.kendall.load_state_dict(checkpoint['kendall_state_dict'])
            print(self.kendall, list(self.kendall.parameters()))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        print('eval epoch {}, training time'.format(epoch, training_time))

        ####################
        # compute_val_loss
        ####################
        torch.cuda.synchronize()
        start = time.time()
        eval_dict = {}
        self.model.eval()
        print('start validation')
        steps = 0
        sum_val_loss = 0
        sum_rec_loss = 0
        sum_sem_loss = 0
        sum_val_acc = 0
        sum_on_acc = 0
        sum_off_acc = 0
        self.iou_cal = IoUCalculator(self.opt)
        self.val_dataset.batch_size = 1
        self.val_data_loader = self.val_dataset.get_loader()
        num_batches = len(self.val_data_loader)

        with torch.no_grad():
            for val_batch in self.val_data_loader:
                steps += 1
                results = compute_loss(self.model, val_batch, self.opt, 'val', self.iou_cal, self.kendall)
                print("Steps {}/{} validation results:".format(steps, len(self.val_data_loader)), val_batch.get('path'), results)
                sum_val_loss += results[0].item()
                sum_rec_loss += results[1].item()
                sum_sem_loss += results[2].item()
                sum_val_acc += results[3]
                sum_on_acc += results[4]
                sum_off_acc += results[5]
                eval_dict[val_batch.get('path')[0]] = results

        val_loss = sum_val_loss / num_batches
        val_acc = sum_val_acc / num_batches
        val_rec_loss = sum_rec_loss / num_batches
        val_sem_loss = sum_sem_loss / num_batches
        val_on_acc = sum_on_acc / num_batches
        val_off_acc = sum_off_acc / num_batches

        if 'syn' in self.opt.exp_name or 'rec' in self.opt.task:
                    mean_iou=0.0
                    overall_acc=0.0
                    mean_acc=0.0
                    iou_list=[0.0]
                    acc_list=[0.0]
        else:
                    mean_iou, overall_acc, mean_acc, iou_list, acc_list = self.iou_cal.compute_iou()

        print('val loss batch avg: ', val_loss)
        print('val rec_loss batch avg: ', val_rec_loss, self.opt.num_sample_points_training, val_rec_loss / self.opt.num_sample_points_training)
        print('val sem_loss batch avg: ', val_sem_loss)
        print('val on acc batch avg: ', val_on_acc)
        print('val off acc batch avg: ', val_off_acc)
        print('mean iou: ', mean_iou)
        print('mean acc: ', mean_acc)
        print('overall acc: ', overall_acc)
        print(iou_list)
        print(acc_list)
        print('End of validation {}'.format(epoch))
        torch.cuda.synchronize()
        print("duration:", time.time() - start)
        with open('eval-{}.pkl'.format(epoch), 'wb') as f:
            pickle.dump(eval_dict, f)


    def save_checkpoint(self, epoch, training_time,last=True):
        path = self.checkpoint_path + 'checkpoint_{}.tar'.format('latest' if last else 'best')
        print('save to ', path)
        if self.kendall is not None:
                torch.save({
                            'training_time': training_time ,'epoch':epoch,
                            'model_state_dict': self.model.state_dict(),
                            'kendall_state_dict': self.kendall.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()}, path)
        else:
                torch.save({ 
                            'training_time': training_time ,'epoch':epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()}, path)



    def load_checkpoint(self):

        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0,0
        path = self.checkpoint_path + 'checkpoint_{}.tar'.format('best' if self.checkpoint else 'latest' )

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.kendall is not None:
            self.kendall.load_state_dict(checkpoint['kendall_state_dict'])
            print(self.kendall, list(self.kendall.parameters()))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        return epoch, training_time


    def compute_val_loss(self):
        self.model.eval()
        print('start validation')
        sum_val_loss = 0
        sum_rec_loss = 0
        sum_sem_loss = 0
        sum_val_acc = 0
        sum_on_acc = 0
        sum_off_acc = 0
        # num_batches = 312
        self.val_dataset.batch_size = 1
        num_batches = self.val_dataset.__len__()
        self.iou_cal = IoUCalculator(self.opt)
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    val_batch = next(self.val_data_iterator.next())
                except:
                    self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                    val_batch = next(self.val_data_iterator)

                results = compute_loss(self.model, val_batch, self.opt, 'val', self.iou_cal, self.kendall)
                print("validation results:", val_batch.get('path'), results)
                sum_val_loss += results[0].item()
                sum_rec_loss += results[1].item()
                sum_sem_loss += results[2].item()
                sum_val_acc += results[3]
                sum_on_acc += results[4]
                sum_off_acc += results[5]

        return sum_val_loss / num_batches,  sum_val_acc / num_batches, sum_rec_loss / num_batches, sum_sem_loss / num_batches, sum_on_acc / num_batches, sum_off_acc / num_batches


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds
