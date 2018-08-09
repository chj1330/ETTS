"""Getting scores.

usage: conversion_one.py [options]

options:
    --data_root=<dir>            Directory contains preprocessed features [default: ../data].
    --checkpoint_path=<path>     Restore model from checkpoint path if given.
    --train_dir=<path>           Directory contains preprocessed dtw features.
    --result_dir=<path>          Result directory.
    -h, --help                   Show this help message and exit
"""
import os
import numpy as np
from os.path import join

from model import EVCModel, NEU2EMO, MEL2LIN, GST
import torch
import audio
from docopt import docopt
import torch.backends.cudnn as cudnn
from util import norm_minmax
from hparams import hparams
from train import build_model, load_checkpoint


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)

    data_root = args["--data_root"]
    checkpoint_path = args["--checkpoint_path"]
    train_dir = args["--train_dir"]
    # Which model to be trained
    result_dir = args["--result_dir"]
    if train_dir is None:
        train_dir = '../feat'  # ./data/feat
    if result_dir is None:
        ling = 'ema00027'
        result_dir = join('./refer_token', ling)  # ./data/feat
    checkpoint_path = './checkpoint/AM_token/checkpoint_epoch000000200_best.pth'

    os.makedirs(result_dir, exist_ok=True)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    train_seq2seq = True
    train_postnet = True
    model = build_model(train_seq2seq, train_postnet, device, hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.initial_learning_rate, betas=(hparams.adam_beta1, hparams.adam_beta2),
                                 eps=hparams.adam_eps, weight_decay=hparams.weight_decay, amsgrad=False)
    if checkpoint_path is not None:
        model, optimizer, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    model.eval()
    LING_DIR = join(train_dir, 'Linguistic_frame')
    MEL_DIR = join(train_dir, 'Acoustic_frame/mel')
    LINEAR_DIR = join(train_dir, 'Acoustic_frame/linear')
    ling_name = ling+'.npy'
    ling = np.load(join(LING_DIR, ling_name))
    ling = norm_minmax(ling, np.load(join(train_dir, 'stat_linguistic_frame.npy')))
    ling = torch.from_numpy(ling).unsqueeze(0).to(device)
    speaker_list = ['ema', 'emb', 'emc', 'emd', 'eme']
    emotions = [0, 1, 2, 3]

    for ref_spk in speaker_list:
        for emo in emotions:
            spk_emo = '{}00{}27.npy'.format(ref_spk, str(emo))
            mel = np.load(join(MEL_DIR, spk_emo))
            mel = torch.from_numpy(mel).unsqueeze(0).to(device)
            _, _, linear_output = model(ling, mel)

            linear_output = linear_output[0].data.cpu().numpy()
            signal = audio.inv_spectrogram(linear_output.T)
            signal /= np.max(np.abs(signal))
            path = join(result_dir, spk_emo.replace('.npy', '.wav'))
            audio.save_wav(signal, path)

            linear = np.load(join(LINEAR_DIR, spk_emo))
            signal = audio.inv_spectrogram(linear.T)
            signal /= np.max(np.abs(signal))
            path = join(result_dir, spk_emo.replace('.npy', '_refer.wav'))
            audio.save_wav(signal, path)

            print('%s' % spk_emo)




