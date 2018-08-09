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
from docopt import docopt
import numpy as np
from os.path import join
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from torch.utils.data import DataLoader
import torch
import audio
import argparse
import torch.backends.cudnn as cudnn
from util import norm_minmax
from hparams import hparams
from train import build_model, load_checkpoint, PyTorchDataset, collate_fn
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

class _NPYDataSource(FileDataSource):
    def __init__(self, train_dir, col, data_type):
        self.train_dir = train_dir
        self.col = col
        self.frame_lengths = []
        self.data_type = data_type
        self.stat = np.load(join(train_dir, 'stat_linguistic_frame.npy'))
        self.multi_speaker = False
        self.speaker_id = None

    def collect_files(self):
        meta = join(self.train_dir, "{}.txt".format(self.data_type))
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        self.multi_speaker = len(l) == 5
        self.frame_lengths = list(map(lambda l: int(l.decode("utf-8").split("|")[3]), lines))

        paths = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.train_dir, f), paths))
        self.paths = paths
        if self.col == 1 :
            if self.multi_speaker:
                self.speaker_id = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
                return paths, self.speaker_id
        return paths

    def collect_features(self, *args):
        path = args[0]
        if self.col == 1 :
            if self.multi_speaker:
                path, speaker_id = args
                return norm_minmax(np.load(path), self.stat), int(speaker_id)
            else:
                return norm_minmax(np.load(path), self.stat)
        elif self.col == 0:
            return path, np.load(path)
        else:
            return np.load(path)


class LingDataSource(_NPYDataSource):
    def __init__(self, data_root, data_type):
        super(LingDataSource, self).__init__(data_root, 1, data_type)

class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, data_type):
        super(MelSpecDataSource, self).__init__(data_root, 2, data_type)

class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, data_type):
        super(LinearSpecDataSource, self).__init__(data_root, 0, data_type)


class PyTorchDataset(object):
    def __init__(self, Ling, Mel, Linear):
        self.Ling = Ling
        self.Mel = Mel
        self.Linear = Linear
        self.multi_speaker = Ling.file_data_source.multi_speaker
    def __getitem__(self, idx):
        if self.multi_speaker:
            Ling, speaker_id = self.Ling[idx]
            return Ling, self.Mel[idx], self.Linear[idx], speaker_id
        else:
            paths, Linear = self.Linear[idx]
            return self.Ling[idx], self.Mel[idx], Linear, paths,

    def __len__(self):
        return len(self.Mel)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x

def collate_fn(batch):
    lengths = [len(x[0]) for x in batch]
    multi_speaker = len(batch[0]) == 5
    max_length = max(lengths)

    Ling = np.array([_pad_2d(x[0], max_length) for x in batch], dtype=np.float32)
    Lingbatch = torch.FloatTensor(Ling)

    Mel = np.array([_pad_2d(x[1], max_length) for x in batch], dtype=np.float32)
    Melbatch = torch.FloatTensor(Mel)

    Linear = np.array([_pad_2d(x[2], max_length) for x in batch], dtype=np.float32)
    Linearbatch = torch.FloatTensor(Linear)

    lengths = torch.LongTensor(lengths)

    if multi_speaker:
        speaker_ids = torch.LongTensor([x[3] for x in batch])
    else :
        speaker_ids = None
    paths = [x[3] for x in batch]
    return Lingbatch, Melbatch, Linearbatch, paths


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
        result_dir = './token_style'  # ./data/feat
    checkpoint_path = 'checkpoint/AM_token/checkpoint_epoch000000200_best.pth'

    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_seq2seq = True
    train_postnet = True
    model = build_model(train_seq2seq, train_postnet, device, hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.initial_learning_rate, betas=(hparams.adam_beta1, hparams.adam_beta2),
                                 eps=hparams.adam_eps, weight_decay=hparams.weight_decay, amsgrad=False)
    if checkpoint_path is not None:
        model, optimizer, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    text_file = 'plot'
    Ling = FileSourceDataset(LingDataSource(train_dir, text_file))
    Mel = FileSourceDataset(MelSpecDataSource(train_dir, text_file))
    Linear = FileSourceDataset(LinearSpecDataSource(train_dir, text_file))

    Dataset = PyTorchDataset(Ling, Mel, Linear)
    Data_loader = DataLoader(Dataset, batch_size=1, num_workers=hparams.num_workers, collate_fn=collate_fn)
    style_embedding_list = []
    label_list = []

    speaker_list = ['ema', 'emb', 'emc', 'emd', 'eme']
    emotions = [0, 1, 2, 3]

    for step, (ling, mel, _, paths) in enumerate(Data_loader):
        parts = paths[0].strip().split('/')
        file_name = parts[-1]
        model.eval()
        """
        for spk in speaker_list:
            if file_name.count(spk) > 0:
                spk_index = speaker_list.index(spk)
        for emo in emotions :
            if spk_index == 5:
                if int(file_name[4]) < 3:
                    emo_index = 0
                else :
                    emo_index = 1
            elif emo == int(file_name[5]):
                emo_index = emo

        label_index = spk_index * len(emotions) + emo_index
        label_list.append(label_index)
        """
        if 0 < int(file_name[5:8]) < 101:
            index = 0
        elif 100 < int(file_name[5:8]) < 201:
            index = 1
        elif 200 < int(file_name[5:8]) < 301:
            index = 2
        elif 300 < int(file_name[5:8]) < 401:
            index = 3

        label_list.append(index)

        if train_seq2seq:
            ling = ling.to(device)
            mel = mel.to(device)

        with torch.no_grad():
            # Apply model
            style_embedding, _, _ = model(ling, mel)
            style_embedding = style_embedding[0][0].data.cpu().numpy()
            style_embedding_list.append(style_embedding)

            #np.save(join(result_dir, file_name), style_embedding)
            print('%s' % file_name)

    np.save(join(result_dir, 'embed_list.npy'), style_embedding_list)
    np.save(join(result_dir, 'label_list.npy'), label_list)











