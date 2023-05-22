import torch
from torch.autograd import Variable
import torch.optim as optim
from models import Generator, Discriminator
import numpy as np
from init import GP_LAMBDA, LATENT_DIM, BATCH_SIZE, NUM_DISC_UPDATES, device, LOSS, SUR_CONSTANT, SUR_METRIC, device
from torch import nn


def get_val_metric_torch(img):
    assert img.ndim == 3  # [bs, 8, 16]
    img = torch.where(img < 0, 0, img)
    img_n = img / img.sum(axis=(1, 2)).reshape(-1, 1, 1)
     
    idx1 = torch.arange(img.shape[1]).reshape(1, -1, 1).to(device)
    idx2 = torch.arange(img.shape[2]).reshape(1, 1, -1).to(device)

    mu1 = (img_n * idx1).sum(axis=(1, 2))
    mu2 = (img_n * idx2).sum(axis=(1, 2))

    cov11 = (img_n * (idx1 ** 2)).sum(axis=(1, 2)) - mu1 ** 2
    cov12 = (img_n * idx1 * idx2).sum(axis=(1, 2)) - mu1 * mu2
    cov22 = (img_n * (idx2 ** 2)).sum(axis=(1, 2)) - mu2 ** 2

    return torch.cat((mu1.reshape(-1, 1), mu2.reshape(-1, 1), cov11.reshape(-1, 1), cov22.reshape(-1, 1), cov12.reshape(-1, 1), img.sum(axis=(1, 2)).reshape(-1, 1)), dim=1)


class Model:
    def __init__(self):
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.disc_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=0.0001)
        self.gen_optimizer = optim.RMSprop(self.generator.parameters(), lr=0.0001)
        self.disc_scheduler = optim.lr_scheduler.ExponentialLR(self.disc_optimizer, gamma=0.999)
        self.gen_scheduler = optim.lr_scheduler.ExponentialLR(self.gen_optimizer, gamma=0.999)
        self.surrogate_loss = nn.MSELoss()

    def make_fake(self, features):
        bs = features.shape[0]
        data = Variable(torch.rand((bs, LATENT_DIM), dtype=torch.float32), requires_grad=True).to(device)
        return self.generator(features, data)

    def disc_loss(self, d_real, d_fake):
        return torch.mean(d_fake - d_real)

    def disc_hinge_loss(self, d_real, d_fake):
        return nn.ReLU()(1.0 - d_real).mean() + nn.ReLU()(1.0 + d_fake).mean()

    def gen_loss(self, d_real, d_fake):
        return torch.mean(d_real - d_fake)

    def gradient_penalty(self, features, real, fake):
        bs = real.shape[0]
        alpha = torch.rand((bs, 1, 1), requires_grad=True, device=device)
        interpolates = alpha * real + (1 - alpha) * fake
        d_int = self.discriminator(features, interpolates)[0]
        grads = torch.autograd.grad(d_int.sum(), interpolates, create_graph=True, retain_graph=True)[0].reshape(
            (bs, -1))
        # return torch.mean(torch.square(grads.norm(2, dim=1) - 1))
        return torch.mean(torch.maximum(torch.norm(grads, p=2, dim=-1) - 1,
                                        torch.tensor([0], dtype=torch.float32, requires_grad=True).to(device)) ** 2)

    def disc_step(self, feature_batch, data_batch):
        # feature_batch = Variable(torch.from_numpy(feature_batch), requires_grad=True).to(device)
        # data_batch = Variable(torch.from_numpy(data_batch), requires_grad=True).to(device)
        data_fake = self.make_fake(feature_batch)

        self.disc_optimizer.zero_grad()

        d_real, sur_metric_real = self.discriminator(feature_batch, data_batch)
        d_fake, sur_metric_fake = self.discriminator(feature_batch, data_fake)
        if LOSS == 'hinge':
            d_loss = self.disc_hinge_loss(d_real, d_fake)
        elif LOSS == 'simple':
            d_loss = self.disc_loss(d_real, d_fake)
        else:
            d_loss = self.disc_loss(d_real, d_fake) + GP_LAMBDA * self.gradient_penalty(feature_batch, data_batch, data_fake)

        true_sur_metric = get_val_metric_torch(data_fake)[:,SUR_METRIC]
        sur_loss = self.surrogate_loss(sur_metric_fake.squeeze(), true_sur_metric)
        d_loss += SUR_CONSTANT * sur_loss
            
        d_loss.backward()
        self.disc_optimizer.step()

        with torch.no_grad():
            g_loss = self.gen_loss(d_real, d_fake)

        loss_vals = d_loss.detach().item() - SUR_CONSTANT * sur_loss.detach().item(), g_loss.detach().item(), sur_loss.detach().item()
        return loss_vals

    def gen_step(self, feature_batch, data_batch):
        # feature_batch = Variable(torch.from_numpy(feature_batch), requires_grad=True).to(device)
        # data_batch = Variable(torch.from_numpy(data_batch), requires_grad=True).to(device)
        d_real, _ = self.discriminator(feature_batch, data_batch)

        self.gen_optimizer.zero_grad()
        data_fake = self.make_fake(feature_batch)
        d_fake, _ = self.discriminator(feature_batch, data_fake)
        g_loss = self.gen_loss(d_real, d_fake)
        g_loss.backward()
        self.gen_optimizer.step()

        if LOSS == 'hinge':
            d_loss = self.disc_hinge_loss(d_real, d_fake)
        elif LOSS == 'simple':
            d_loss = self.disc_loss(d_real, d_fake)
        else:
            d_loss = self.disc_loss(d_real, d_fake) + GP_LAMBDA * self.gradient_penalty(feature_batch, data_batch, data_fake)

        loss_vals = d_loss.detach().item(), g_loss.detach().item()

        return loss_vals

    def train_epoch(self, data, features, epoch):
        shuffle_ids = np.random.permutation(len(data))
        self.discriminator.train()
        self.generator.train()
        losses = []
        for i in range(0, len(data), BATCH_SIZE):
            data_batch = data[shuffle_ids][i:i + BATCH_SIZE]
            feature_batch = features[shuffle_ids][i:i + BATCH_SIZE]
            d_mean_loss = 0
            sur_mean_loss = 0
            for _ in range(NUM_DISC_UPDATES):
                d_loss, _, sur_loss = self.disc_step(feature_batch, data_batch)
                d_mean_loss += d_loss
                sur_mean_loss += sur_loss
            d_mean_loss /= NUM_DISC_UPDATES
            sur_mean_loss /= NUM_DISC_UPDATES
            _, g_loss = self.gen_step(feature_batch, data_batch)
            losses.append((epoch, i, d_mean_loss, g_loss, sur_mean_loss))
        return losses

    def test_epoch(self, data, features, epoch):
        self.discriminator.eval()
        self.generator.eval()
        losses = np.array([0, 0, 0], dtype='float32')
        for i in range(0, len(data), BATCH_SIZE):
            feature_batch = features[i:i + BATCH_SIZE]
            data_batch = data[i:i + BATCH_SIZE]
            # feature_batch = Variable(torch.from_numpy(features[i:i + BATCH_SIZE]), requires_grad=True).to(device)
            # data_batch = Variable(torch.from_numpy(data[i:i + BATCH_SIZE]), requires_grad=True).to(device)
            # feature_batch = features[i:i + BATCH_SIZE]
            # data_batch = data[i:i + BATCH_SIZE]
            data_fake = self.make_fake(feature_batch)
            d_real, sur_metric_real = self.discriminator(feature_batch, data_batch)
            d_fake, sur_metric_fake = self.discriminator(feature_batch, data_fake)
            if LOSS == 'hinge':
                d_loss = self.disc_hinge_loss(d_real, d_fake)
            elif LOSS == 'simple':
                d_loss = self.disc_loss(d_real, d_fake)
            else:
                d_loss = self.disc_loss(d_real, d_fake) + GP_LAMBDA * self.gradient_penalty(feature_batch, data_batch, data_fake)
            
            true_sur_metric = get_val_metric_torch(torch.cat((data_batch, data_fake)))[:,SUR_METRIC]
            sur_loss = self.surrogate_loss(torch.cat((sur_metric_real, sur_metric_fake)).squeeze(), true_sur_metric)
            
            g_loss = self.gen_loss(d_real, d_fake)
            losses += np.array([d_loss.item(), g_loss.item(), sur_loss.item()])
        losses /= (len(data) / BATCH_SIZE)
        return epoch, losses[0], losses[1], losses[2]

    def scheduler_step(self):
        self.disc_scheduler.step()
        self.gen_scheduler.step()
