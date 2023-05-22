import torch
import os

n_classes = 10
in_channels = 3
out_channels = 32

LATENT_DIM = 32
GP_LAMBDA = 10
BATCH_SIZE = 96
NUM_DISC_UPDATES = 8
NUM_EPOCHS = 10000
SAVE_TIME = 50
L_CONSTANT = 1
SUR_CONSTANT = 1
SUR_METRIC = 4

SPEC_NORM_DISC = False
SPEC_NORM_GEN = False

CONTINUE_TRAIN = False

LOSS = 'wasserstein'  # wasserstein or hinge or simple

pad_range = (-3, 5)
time_range = (-7, 9)

metric_names = ['Mean0', 'Mean1', 'Sigma0^2', 'Sigma1^2', 'Cov01']

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

CHI_FILE = 'evaluation/chi_metric/chi_metric.npy'
CHECKPOINT_PATH = ''
CHI_LOAD_FILE = ''
CHI_SEP_LOAD_FILE = ''


def make_dirs():
    os.mkdir('models')
    os.mkdir('evaluation')
    os.mkdir('evaluation/losses')
    os.mkdir('evaluation/gaus_metrics')
    os.mkdir('evaluation/imgs')
    os.mkdir('evaluation/chi_plots')
    os.mkdir('evaluation/chi_metric')
    os.mkdir('evaluation/chi_metric_sep')
    os.mkdir('evaluation/chi_full_plots')
