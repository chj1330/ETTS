"""Getting scores.

usage: get_scores.py [options]

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
from docopt import docopt
import torch
import audio
from hparams import hparams
from util import norm_minmax
import torch.backends.cudnn as cudnn
from train import build_model, load_checkpoint, PyTorchDataset, collate_fn
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


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
        result_dir = './AM_notoken_27'  # ./data/feat
        mel_result_dir = join(result_dir, 'mel')
    checkpoint_path = 'checkpoint/AM_notoken/checkpoint_epoch000000050_best.pth'

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(mel_result_dir, exist_ok=True)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_seq2seq = True
    train_postnet = True
    model = build_model(train_seq2seq, train_postnet, device, hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.initial_learning_rate, betas=(hparams.adam_beta1, hparams.adam_beta2),
                                 eps=hparams.adam_eps, weight_decay=hparams.weight_decay, amsgrad=False)
    if checkpoint_path is not None:
        model, optimizer, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    model.eval()

    stat = np.load(join(train_dir, 'stat_linguistic_frame.npy'))
    with open(join(train_dir, 'test.txt'), encoding='utf-8') as f:
        for line in f:
            fn = line.split()
            l = fn[0].split("|")
            ling = norm_minmax(np.load(join(train_dir,l[1])), stat)
            mel = np.load(join(train_dir, l[2]))
            file_name = l[1].strip().split('/')
            file_name = file_name[-1]
            ling = torch.from_numpy(ling).unsqueeze(0).to(device)
            mel = torch.from_numpy(mel).unsqueeze(0).to(device)
            with torch.no_grad():
                _, mel_outputs, linear_outputs = model(ling, mel)

            mel_output = mel_outputs[0].data.cpu().numpy()
            path = join(mel_result_dir, file_name)
            np.save(path, mel_output)


            linear_output = linear_outputs[0].data.cpu().numpy()
            signal = audio.inv_spectrogram(linear_output.T)
            signal /= np.max(np.abs(signal))
            path = join(result_dir, file_name.replace('.npy', '.wav'))
            audio.save_wav(signal, path)

            linear = np.load(join(train_dir,l[0]))
            signal = audio.inv_spectrogram(linear.T)
            signal /= np.max(np.abs(signal))
            path = join(result_dir, file_name.replace('.npy', '_refer.wav'))
            audio.save_wav(signal, path)


            print('%s' % file_name)


