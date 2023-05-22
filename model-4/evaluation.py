import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import torch
import os
from init import pad_range, time_range, BATCH_SIZE, device, metric_names, SAVE_TIME, CHI_FILE
from trends import make_trend_plot
from combine_imgs import combine_imgs


def show_losses(losses_train, losses_test, n, epoch):
    losses_train = pd.DataFrame(losses_train, columns=['epoch', 'sample', 'disc_loss', 'gen_loss', 'sur_loss'])
    losses_test = pd.DataFrame(losses_test, columns=['epoch', 'disc_loss', 'gen_loss', 'sur_loss'])
    plt.figure()
    plt.plot(losses_train.epoch + losses_train['sample'] / n, losses_train.disc_loss, label='disc, train')
    plt.plot(losses_test.epoch, losses_test.disc_loss, label='disc, test')
    plt.plot(losses_train.epoch + losses_train['sample'] / n, losses_train.gen_loss, label='gen, train')
    plt.plot(losses_test.epoch, losses_test.gen_loss, label='gen, test')
    plt.plot(losses_train.epoch + losses_train['sample'] / n, losses_train.sur_loss, label='sur, train')
    plt.plot(losses_test.epoch, losses_test.gen_loss, label='sur, test')
    plt.legend()
    plt.savefig('evaluation/losses/' + str(epoch) + '.png')


def gaussian_fit(img):
    assert img.ndim == 2
    assert (img >= 0).all()
    assert (img > 0).any()
    img_n = img / img.sum()

    mu = np.fromfunction(
        lambda i, j: (img_n[np.newaxis,...] * np.stack([i, j])).sum(axis=(1, 2)),
        shape=img.shape
    )
    cov = np.fromfunction(
        lambda i, j: (
            (img_n[np.newaxis,...] * np.stack([i * i, j * i, i * j, j * j])).sum(axis=(1, 2))
        ) - np.stack([mu[0]**2, mu[0]*mu[1], mu[0]*mu[1], mu[1]**2]),
        shape=img.shape
    ).reshape(2, 2)

    return mu, cov


def get_val_metric_single(img):
    assert img.ndim == 2
    img = np.where(img < 0, 0, img)
    mu, cov = gaussian_fit(img)
    # print(img)
    # print(mu)
    # print(cov)
    return np.array((*mu, *cov.diagonal(), cov[0, 1], img.sum()))


get_val_metric = np.vectorize(get_val_metric_single, signature='(m,n)->(k)')


def show_metrics(metric_real, metric_fake, epoch):
    # metric_real = get_val_metric(real_data)
    # metric_fake = get_val_metric(fake_data)
    plt.figure(figsize=(14, 8))
    labels = ['x0', 'x1', 'sigma0^2', 'sigma1^2', 'cov', 'amplitude']
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        r = metric_real[:, i]
        f = metric_fake[:, i]
        bins = np.linspace(
            min(r.min(), f.min()),
            max(r.max(), f.max()),
            50
        )
        plt.hist(r, bins=bins, label='real', alpha=0.8)
        plt.hist(f, bins=bins, label='generated', alpha=0.5)
        plt.legend()
        plt.title(labels[i])
    plt.tight_layout()
    plt.savefig('evaluation/gaus_metrics/' + str(epoch) + '.png')


def show_imgs(real_data, fake_data, epoch):
    plt.figure(figsize=(17, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(fake_data[i, ::-1], extent=(*time_range, *pad_range))
        plt.title('generated')

    for i in range(5):
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(real_data[i, ::-1], extent=(*time_range, *pad_range))
        plt.title('real')
    plt.savefig('evaluation/imgs/' + str(epoch) + '.png')


def buf_to_file(buf, filename):
    with open(filename, 'wb') as f:
        f.write(buf.getbuffer())


def get_metric(features_real, features_fake, metrics_real, metrics_fake, prev_chi_metrics, prev_chi_metrics_sep, epoch):
    features = {
        'crossing_angle': (features_real[:, 0], features_fake[:, 0]),
        'dip_angle': (features_real[:, 1], features_fake[:, 1]),
        'drift_length': (features_real[:, 2], features_fake[:, 2]),
        'time_bin_fraction': (features_real[:, 2] % 1, features_fake[:, 2] % 1),
        'pad_coord_fraction': (features_real[:, 3] % 1, features_fake[:, 3] % 1)
    }
    plots = {}
    chi = [0] * 6
    for feature_name, (feature_real, feature_gen) in features.items():
        for i, metric_name in enumerate(metric_names):
            name = f'{metric_name} vs {feature_name}'
            pngfile = io.BytesIO()
            plots[name] = pngfile
            _, cur_chi = make_trend_plot(feature_real, metrics_real[:, i], feature_gen,
                                                          metrics_fake[:, i], '', calc_chi2=True, pngfile=pngfile)
            chi[i] += cur_chi

    os.mkdir('evaluation/chi_plots/' + str(epoch))
    for k, img in plots.items():
        buf_to_file(img, 'evaluation/chi_plots/' + str(epoch) + '/' + str(k) + '.png')

    chi_metric = sum(chi)
    prev_chi_metrics_sep += [chi]
    with open('evaluation/chi_metric_sep/chi_metric_sep.npy', 'wb') as f:
        np.save(f, prev_chi_metrics_sep)

    plt.figure()
    plt.plot(np.arange(0, (len(prev_chi_metrics) + 1) * SAVE_TIME, SAVE_TIME), prev_chi_metrics + [chi_metric])
    plt.savefig('evaluation/chi_metric/' + str(epoch) + '.png')
    return chi_metric


def evaluate_model(gen_model, features_real, data_real, losses_train, losses_test, k, prev_chi_metrics, prev_chi_metrics_sep, epoch, gen_more=5):
    if gen_more:
        features_fake = np.tile(features_real, [gen_more] + [1] * (features_real.ndim - 1))
    else:
        features_fake = features_real
    data_fake = np.concatenate([
        gen_model.make_fake(torch.from_numpy(features_fake[i:i+BATCH_SIZE]).to(device)).cpu().detach().numpy()
        for i in range(0, len(features_fake), BATCH_SIZE)
    ], axis=0)
    data_fake[data_fake < 0] = 0
    metrics_real = get_val_metric(data_real)
    metrics_fake = get_val_metric(data_fake)
    show_metrics(metrics_real, metrics_fake[:len(data_real)], epoch)
    show_imgs(data_real, data_fake[:len(data_real)], epoch)
    show_losses(losses_train, losses_test, k, epoch)
    chi = get_metric(features_real, features_fake, metrics_real, metrics_fake, prev_chi_metrics,  prev_chi_metrics_sep, epoch)

    with open(CHI_FILE, 'wb') as f:
        np.save(f, prev_chi_metrics + [chi])

    combine_imgs(epoch)

    return chi
