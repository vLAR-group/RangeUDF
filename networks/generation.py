import torch
import os
from glob import glob
import numpy as np
from torch.nn import functional as F
import time
import nearest_neighbors

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)

class Generator(object):
    def __init__(self, model, opt, exp_name, log_dir, threshold = 0.1, checkpoint = -1, device = torch.device("cuda")):
        self.checkpoint = checkpoint
        model = torch.nn.DataParallel(model)
        self.model = model.to("cuda:0")
        self.model.eval()
        self.opt = opt

        if log_dir is None:
            self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format(exp_name)
        else:
            self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/{}/checkpoints/'.format(exp_name, log_dir)

        self.load_checkpoint()
        self.threshold = threshold


    def generate_point_cloud(self, data, num_steps = 10, num_points = 1600000, filter_val = 0.009):

        # get encoder input
        feature = data.get('feature').cuda()
        on_surface_points = [item.to("cuda:0", non_blocking=True) for item in data.get('on_surface_points')]
        neigh_idx = [item.to("cuda:0", non_blocking=True) for item in data.get('input_neighbors')]
        sub_idx = [item.to("cuda:0", non_blocking=True) for item in data.get('input_pools')]
        on_interp_idx = [item.to("cuda:0", non_blocking=True) for item in data.get('on_interp_idx')]
        num_on_surface = len(on_surface_points[0][0])

        encoding = self.model.module.backbone(on_surface_points, feature, neigh_idx, sub_idx, on_interp_idx) # out -> feature, features_on, coords

        for param in self.model.module.parameters():
            param.requires_grad = False

        sample_num = 200000
        samples_cpu = np.zeros((0, 3))
        samples = torch.rand(1, sample_num, 3).float().to("cuda:0") - 0.5
        samples.requires_grad = True

        i = 0
        start = time.time()
        while len(samples_cpu) < num_points:
            print('iteration', i)

            for j in range(num_steps):
                print('refinement', j)
                interp_idx = nearest_neighbors.knn(on_surface_points[0][0].cpu().numpy(), samples[0].detach().cpu().numpy(), self.opt.num_interp, omp=True) # encoding[2][0] -> coords[0]: (B, N, 3)
                interp_idx = torch.from_numpy(interp_idx).unsqueeze(0).cuda()

                df_pred, _ = self.model.module.task([interp_idx], samples, *encoding)
                
                df_pred = torch.clamp(df_pred, max=self.threshold)
                df_pred.sum().backward(retain_graph=True)

                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
            
                feature = feature.detach()
                samples = samples - F.normalize(gradient, dim=2) * df_pred[:, num_on_surface:].reshape(-1, 1)  # better use Tensor.copy method?
                # samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)
                samples = samples.detach()
                samples.requires_grad = True

            print('finished refinement')
            # bounds cut
            # a = ((samples<-0.5) | (samples>0.5)).cpu().numpy()
            # outlier = np.unique(np.where(a)[1])
            # cond = np.delete(np.arange(0, sample_num), outlier)

            # off surface generation
            df_pred = df_pred[:, num_on_surface:]
            if not i == 0:
                # samples_cpu = np.vstack((samples_cpu, samples[:, cond][df_pred[:, cond] < filter_val].detach().cpu().numpy()))
                samples_cpu=np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))

            samples = samples[df_pred < 0.03].unsqueeze(0)
            indices = torch.randint(samples.shape[1], (1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (self.threshold / 3) * torch.randn(samples.shape).to("cuda:0")  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            print(samples_cpu.shape)

        duration = time.time() - start
        return data['on_surface_points'][0][0], samples_cpu,  duration



    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0,0

        path = self.checkpoint_path + 'checkpoint_{}.tar'.format('best' if self.checkpoint else 'latest')

        print('Loaded checkpoint from: {}'.format(path))

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        print('eval epoch {}, training time {}'.format(epoch, training_time))
        return epoch, training_time


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
