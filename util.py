from __future__ import division, print_function, absolute_import
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import audio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing.data import _handle_zeros_in_scale
from sklearn.utils.extmath import _incremental_mean_and_var


def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def masked_mean(y, mask):
    # (B, T, D)
    mask_ = mask.expand_as(y)
    return (y * mask_).sum() / mask_.sum()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()

def prepare_spec_image(spectrogram):
    # [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    return np.uint8(cm.magma(spectrogram.T) * 255)



def norm_meanvar(x,stat):
    return scale(x, stat[2], stat[3])

def norm_minmax(x,stat):
    x = norm_meanvar(x,stat)
    return  minmax_scale(x, min_=stat[4], scale_=stat[5])

def denorm_meanvar(x, stat):
    return inv_scale(x, stat[2], stat[3])

def denorm_minmax(x,stat):
    x = inv_minmax_scale(x, min_=stat[4], scale_=stat[5])
    return denorm_meanvar(x, stat)

def stat(x):
    feat_min, feat_max = minmax(x)
    feat_mean, feat_std = meanstd(x)
    feat_min = scale(feat_min, feat_mean, feat_std)
    feat_max = scale(feat_max, feat_mean, feat_std)
    feat_min2, feat_scale = minmax_scale_params(feat_min, feat_max, feature_range=(0.01, 0.99))
    stat_out = np.stack([feat_min, feat_max, feat_mean, feat_std, feat_min2, feat_scale],axis=0)
    return stat_out

def adjust_label_lws(labels, wav_length, fft_size=1024, hop_size=256, sample_rate=22050):
    l, r = audio.lws_pad_lr(wav_length, fft_size, hop_size)
    pad_l = int(l/sample_rate*10000000)
    labels.start_times = [x+pad_l for x in labels.start_times]
    labels.end_times = [x + pad_l for x in labels.end_times]
    labels.start_times[0] = 0
    labels.end_times[-1] = int((wav_length+l+r)/sample_rate*10000000)
    return labels

def adjust_wav_lws(labels, wav_length, fft_size=1024, hop_size=256, sample_rate=22050):
    l, r = audio.lws_pad_lr(wav_length, fft_size, hop_size)
    pad_l = int(l/sample_rate*10000000)
    labels.start_times = [x+pad_l for x in labels.start_times]
    labels.end_times = [x + pad_l for x in labels.end_times]
    labels.start_times[0] = 0
    labels.end_times[-1] = int((wav_length+l+r)/sample_rate*10000000)

def parallelProcessing(functions, ppnum=1):
    if ppnum<0:
        ppnum=multiprocessing.cpu_count()-2
    elif ppnum>multiprocessing.cpu_count():
        ppnum=multiprocessing.cpu_count()
    with ProcessPoolExecutor(ppnum) as executor:
        futures = [executor.submit(functions[idx]) for idx in range (len(functions))]
        for future in tqdm(futures):
            future.result()

def meanvar(dataset):
    last_sample_count=0
    mean_ = 0.
    var_ = 0.
    dtype = dataset[0].dtype

    for idx, x in enumerate(dataset):
        mean_, var_, _ = _incremental_mean_and_var(
            x, mean_, var_, last_sample_count)
        last_sample_count += len(x)
    mean_, var_ = mean_.astype(dtype), var_.astype(dtype)

    return mean_, var_

def meanstd(dataset):
    m, v = meanvar(dataset)
    v = np.sqrt(v)
    return m, v

def minmax(dataset):
    max_ = -np.inf
    min_ = np.inf

    for idx, x in enumerate(dataset):
        min_ = np.minimum(min_, np.min(x, axis=(0,)))
        max_ = np.maximum(max_, np.max(x, axis=(0,)))

    return min_, max_

def scale(x, data_mean, data_std):
    return (x - data_mean) / _handle_zeros_in_scale(data_std, copy=False)


def inv_scale(x, data_mean, data_std):
    return data_std * x + data_mean


def minmax_scale_params(data_min, data_max, feature_range=(0, 1)):
    data_range = data_max - data_min
    scale_ = (feature_range[1] - feature_range[0]) / \
        _handle_zeros_in_scale(data_range, copy=False)
    min_ = feature_range[0] - data_min * scale_
    return min_, scale_

def minmax_scale(x, scale_=None, min_=None):
    return x * scale_ + min_

def inv_minmax_scale(x, scale_=None, min_=None):
    return (x - min_) / scale_
