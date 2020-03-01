import os
from math import sqrt

from utils import *
from torch.optim import Adam
from torchvision.utils import save_image

#def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, A, anneal_power=2.):
#    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
#    perturbed_samples = samples + A*used_sigmas#torch.randn_like(samples) * used_sigmas
#    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
#    scores = scorenet(perturbed_samples, labels)
#    target = target.view(target.shape[0], -1)
#    scores = scores.view(scores.shape[0], -1)
#    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
#
#    return loss.mean(dim=0)

class NCSNTrainer():

    def __init__(self, ncsn, lr, dataloader, num_iters, beta1, beta2, checkpoints_folder, save_every):
        self.L = 10
        self.sigmas = torch.Tensor([10*0.599**i for i in range(0, 10)])
        self.ncsn = ncsn
        self.ncsn_opt = Adam(
                self.ncsn.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.dataloader = dataloader
        self.num_iters = num_iters
        self.save_folder = checkpoints_folder
        self.save_every = save_every


    def _loss_tensor(self, batch, which_sigmas):
        batch_size = batch.shape[0]
        #compute target
        selected_sigmas = torch.index_select(self.sigmas, 0, which_sigmas)
        selected_sigmas = selected_sigmas.view(
                [batch.shape[0]] + [1]*(len(batch.shape)-1))
        perturbed_batch = batch + torch.randn_like(batch)*selected_sigmas
        target = -(perturbed_batch - batch)/(selected_sigmas**2)
        #compute output
        output = self.ncsn(perturbed_batch, which_sigmas)
        #loss is ~ euclidean norm squared of the difference
        diff = target - output
        loss = (1/(2*batch_size)) * torch.sum((diff*selected_sigmas)**2)
        return loss

    
    def train_ncsn(self):
        os.makedirs(self.save_folder, exist_ok=True)
        curr_iter = 0
        while True:
            for batch, __ in self.dataloader:
                self.ncsn_opt.zero_grad()
                #choose random sigma for each image in batch 
                #to minimize expected loss
                which_sigmas = torch.randint(
                        0, len(self.sigmas), (batch.shape[0],))
#                A = torch.randn_like(batch)
                loss = self._loss_tensor(batch, which_sigmas)
#                loss2 = anneal_dsm_score_estimation(self.ncsn, batch, which_sigmas, self.sigmas, A) 
#                print(loss.item(), loss2.item())
                loss.backward()
                self.ncsn_opt.step()
                if curr_iter%10==0:
                    print('iter: ', curr_iter, 
                            '  |  loss: ', round(loss.item(), 3))
                curr_iter += 1
                if curr_iter % self.save_every==0 and curr_iter > 0:
                    self._save_model(curr_iter)
                if curr_iter >= self.num_iters:
                    return


    def _save_model(self, curr_iter):
        save_num = int(curr_iter/self.save_every)
        save_path = self.save_folder + '/ncsn_' + str(save_num)
        torch.save(self.ncsn.state_dict(), save_path) 


    def _get_image_shape(self):
        batch, __ = next(iter(self.dataloader))
        shape = (1, *batch.shape[1:])
        return shape


    def annealed_langevin_dynamics(self, trained_ncsn, eps, T):
        image_shape = self._get_image_shape()
        L = len(self.sigmas)
        x_prev = torch.rand(*image_shape) - 0.5 #uniform
        for i in range(L):
            ai = eps * self.sigmas[i]**2/self.sigmas[L-1]**2
            i_tensor = torch.Tensor([i]).long()
            for t in range(T):
                zt = torch.randn(*image_shape) #normal 0, I
                score = trained_ncsn(x_prev, i_tensor) 
                x_curr = x_prev + score*ai/2 +sqrt(ai)*zt
                x_prev = x_curr
            print('finished sigma_' + str(i))
        return x_curr


    def save_samples(self, num_samples, nrow, trained_ncsn, eps, T):
        root = 'sampled_images'
        os.makedirs(root, exist_ok=True)
        samples = []
        with torch.no_grad():
            for i in range(num_samples):
                sample = self.annealed_langevin_dynamics(trained_ncsn, eps, T)
                samples.append(sample)
        samples = torch.cat(samples)
        save_image(samples, root + '/samples.png', nrow=nrow)
