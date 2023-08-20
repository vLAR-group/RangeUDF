import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_loss(model, batch, opt, mode, iou_cal=None, kendall=None):

    if 'rec' in opt.task or 'rec' in opt.exp_name:
        df_gt = batch.get('df').cuda() #(Batch,num_points)
        if 'joint' in opt.task:
            df_pred, _ = model(batch) #(Batch,num_points), (Batch,num_classes,num_points)
        else:
            df_pred = model(batch)
        rec_loss, rec_on, rec_off = compute_rec_loss(df_pred, df_gt, opt.num_points, opt)
        loss = rec_loss
        acc, on_acc, off_acc = 0, 0, 0
        sem_loss, sem_on, sem_off = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

    elif 'sem' in opt.task or 'sem' in opt.exp_name:
        sem_gt = batch.get('targets').cuda()
        df_gt = batch.get('df').cuda() #(Batch,num_points)
        if 'joint' in opt.task:
            _, sem_pred = model(batch) #(Batch,num_points), (Batch,num_classes,num_points)
        else:
            sem_pred = model(batch)

        rec_loss, rec_on, rec_off = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        sem_loss, sem_on, sem_off = compute_sem_loss(sem_pred, sem_gt, df_gt, opt.num_points, opt)
        loss = sem_loss
        acc, on_acc, off_acc = accuracy(sem_pred, sem_gt, opt, mode, iou_cal)

    elif 'joint' in opt.task or 'joint' in opt.exp_name:
        df_gt = batch.get('df').cuda() #(Batch,num_points)
        sem_gt = batch.get('targets').cuda()
        df_pred, sem_pred = model(batch) #(Batch,num_points), (Batch,num_classes,num_points)

        rec_loss, rec_on, rec_off = compute_rec_loss(df_pred, df_gt, opt.num_points, opt)
        sem_loss, sem_on, sem_off = compute_sem_loss(sem_pred, sem_gt, df_gt, opt.num_points, opt)
        sem_loss, sem_on, sem_off = sem_loss/100, sem_on/100, sem_off/100

        if opt.joint_mode == 'naive' and kendall is None:
            loss = rec_loss + sem_loss
        elif 'kendall' in opt.joint_mode and kendall is not None:
            loss, _, _ = kendall([rec_loss, sem_loss])
        else:
            raise ValueError

        acc, on_acc, off_acc = accuracy(sem_pred, sem_gt, opt, mode, iou_cal)

    else:
        raise ValueError

    return loss, rec_loss, sem_loss, acc, on_acc, off_acc, rec_on, rec_off, sem_on, sem_off


def compute_rec_loss(df_pred, df_gt, num_points, opt):
    try:
            loss_on = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred[:, :num_points], max=opt.max_dist),torch.clamp(df_gt[:, :num_points], max=opt.max_dist))
            loss_off = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred[:, num_points:], max=opt.max_dist),torch.clamp(df_gt[:, num_points:], max=opt.max_dist))
            loss = torch.cat((opt.reg_coef * loss_on, loss_off), dim=-1).sum(-1).mean()
    except:
            raise ValueError
    return loss, opt.reg_coef * loss_on.sum(-1).mean(), loss_off.sum(-1).mean()


def compute_sem_loss(sem_pred, sem_gt, df_gt, num_points, opt):
    if opt.label_mode == 'full':
        balance_factor = 1
    elif opt.label_mode == 'weak1':
        balance_factor = 10
    elif opt.label_mode == 'weak2':
        balance_factor = 100
    elif opt.label_mode == 'weak3':
        balance_factor = 1000
    elif opt.label_mode == 'weak4':
        balance_factor = 2500
    elif opt.label_mode == 'weak5':
        balance_factor = 10000

    loss_on_sem = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(sem_pred[:, :, :num_points], sem_gt[:, :num_points]) * balance_factor
    loss_off_sem = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(sem_pred[:, :, num_points:], sem_gt[:, num_points:]) * balance_factor

    if opt.sem_term == 'off':
        if opt.sem_loss == 'exp':
            loss_off_sem  = loss_off_sem * torch.exp(-100 * df_gt[:, num_points:]) 
            loss_sem = torch.cat((opt.sem_coef * loss_on_sem, loss_off_sem), dim=-1).sum(-1).mean() # / 100
        elif opt.sem_loss == 'ori':
            loss_sem = torch.cat((opt.sem_coef * loss_on_sem, loss_off_sem), dim=-1).sum(-1).mean() # / 100
        else:
            raise ValueError
    else:
        raise ValueError

    return loss_sem, opt.sem_coef * loss_on_sem.sum(-1).mean(), loss_off_sem.sum(-1).mean()


class IoUCalculator:
    def __init__(self, cfg):
        print('initializing new iou calculator')
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg
        self.val_total_correct = 0
        self.val_total_seen = 0

    def add_data(self, pred, labels):

        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        correct = np.sum(pred_valid == labels_valid)
        self.val_total_correct += correct
        self.val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        acc_list = []
        t = 0
        gt = 0
        for n in range(0, self.cfg.num_classes, 1):

            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                assert self.gt_classes[n] != 0
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                t += self.true_positive_classes[n]
                gt += self.gt_classes[n]
                acc = self.true_positive_classes[n] / float(self.gt_classes[n])
                iou_list.append(iou)
                acc_list.append(acc)
            else:
                iou_list.append(0.0)
                acc_list.append(0.0)

        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        mean_acc = sum(acc_list) / float(self.cfg.num_classes)
        if t == 0 or gt ==0:
            t=0
            gt=1
        overall_acc = t / gt
        print(self.gt_classes)
        return mean_iou, overall_acc, mean_acc, iou_list, acc_list


def accuracy(outputs, labels, opt, mode, iou_cal):
    """
    Computes accuracy of the current batch
    :param outputs: logits predicted by the taskwork
    :param labels: labels
    :return: accuracy value
    """

    outputs = outputs.transpose(1, 2)#.reshape(-1, opt.num_classes)
    labels = labels#.reshape(-1)
    target = labels
    valid_bool_idx = labels!=-1

    predicted = torch.argmax(outputs, dim=-1)
    total = len(np.nonzero(valid_bool_idx)) # target.size(0)
    correct = (predicted[valid_bool_idx] == target[valid_bool_idx]).sum().item()

    if (len(np.nonzero(labels[:, :opt.num_points]!=-1))!=0) and (len(np.nonzero(labels[:, opt.num_points:]!=-1))!=0):
        on_acc = (predicted[:, :opt.num_points][labels[:, :opt.num_points]!=-1] == target[:, :opt.num_points][labels[:, :opt.num_points]!=-1]).sum().item() / len(np.nonzero(labels[:, :opt.num_points]!=-1))
        off_acc = (predicted[:, opt.num_points:][labels[:, opt.num_points:]!=-1] == target[:, opt.num_points:][labels[:, opt.num_points:]!=-1]).sum().item() / len(np.nonzero(labels[:, opt.num_points:]!=-1))
    else:
        print('error！no label available')
        on_acc = (predicted[:, :opt.num_points][labels[:, :opt.num_points]!=-1] == target[:, :opt.num_points][labels[:, :opt.num_points]!=-1]).sum().item() / (len(np.nonzero(labels[:, :opt.num_points]!=-1)) + 1)
        off_acc = (predicted[:, opt.num_points:][labels[:, opt.num_points:]!=-1] == target[:, opt.num_points:][labels[:, opt.num_points:]!=-1]).sum().item() / (len(np.nonzero(labels[:, opt.num_points:]!=-1)) + 1)

    # IoU
    if mode == 'val':
        iou_cal.add_data(predicted[valid_bool_idx], target[valid_bool_idx])

    return correct / total, on_acc, off_acc


class kendall1(nn.Module):
    def __init__(self, task_num=2):
        super().__init__()
        self.sigmas = []
        for i in range(task_num):
            weights = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32), requires_grad=True)
            torch.nn.init.uniform_(weights, 0.2, 1)
            self.sigmas.append(weights)
        self.list = nn.ParameterList(self.sigmas)

    def forward(self, loss_list):
        loss_sum = 0
        list = []
        for i , loss in enumerate(loss_list):
            factor = 0.5 / (self.sigmas[i])
            tmp = factor*loss + torch.log(self.sigmas[i])
            loss_sum += tmp
            list.append(tmp)
        return loss_sum, list[0].mean(), list[1].mean()

class kendall2(nn.Module):
    def __init__(self, task_num=2):
        super().__init__()
        self.sigmas = []
        for i in range(task_num):
            weights = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32), requires_grad=True)
            torch.nn.init.uniform_(weights, 0.2, 1)
            self.sigmas.append(weights)
        self.list = nn.ParameterList(self.sigmas)

    def forward(self, loss_list):
        loss_sum = 0
        list = []
        for i , loss in enumerate(loss_list):
            factor = 0.5 / (self.sigmas[i]**2)
            tmp = factor*loss + torch.log(self.sigmas[i]**2)
            loss_sum += tmp
            list.append(tmp)
        return loss_sum, list[0].mean(), list[1].mean()

class kendall3(nn.Module):
    def __init__(self, nb_outputs=2):      
        super().__init__()
        self.log_vars1 = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.log_vars2 = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

    def forward(self, loss_list):
        tmp1 = torch.exp(-self.log_vars1) * loss_list[0] + self.log_vars1
        tmp2 = torch.exp(-self.log_vars2) * loss_list[1] + self.log_vars2
        loss = tmp1 + tmp2 
        
        return loss, tmp1.mean(), tmp2.mean()

class kendall4(nn.Module):
    def __init__(self, nb_outputs=2):      
        super().__init__()
        self.log_vars1 = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.log_vars2 = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        torch.nn.init.uniform_(self.log_vars1, 0.2, 1)
        torch.nn.init.uniform_(self.log_vars2, 0.2, 1)

    def forward(self, loss_list):
        tmp1 = torch.exp(-self.log_vars1) * loss_list[0] + self.log_vars1
        tmp2 = torch.exp(-self.log_vars2) * loss_list[1] + self.log_vars2
        loss = tmp1 + tmp2 
        
        return loss, tmp1.mean(), tmp2.mean()
