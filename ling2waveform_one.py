import os
import numpy as np
from os.path import join
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from model import EVCModel, NEU2EMO, MEL2LIN, GST
import torch
import audio
import argparse
import torch.backends.cudnn as cudnn
from util import norm_meanvar
from hparams import hparams
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

if __name__ == "__main__":
    GPU_USE = 0
    DATA_ROOT = '../data'
    RESULT_DIR = join('./result')
    MEL_DIR = join(DATA_ROOT, 'feat', 'Acoustic_frame', 'mel')  # ./data/feat/Acoustic_frame/mel
    LING_DIR = join(DATA_ROOT, 'feat', 'Linguistic_frame')  # ./data/feat/Linguistic_frame
    #CHECKPOINT = "./checkpoint/Sad/2018-07-17_20:50:58.394217/checkpoint_epoch000006500.pth"
    CHECKPOINT = "./checkpoint/2018-07-25_22:43:32.085533/checkpoint_epoch000002300.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR, help='result directory')
    parser.add_argument('--gpu_use', type=int, default=GPU_USE, help='GPU enable? 0 : cpu, 1 : gpu')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='data directory')
    parser.add_argument('--mel_dir', type=str, default=MEL_DIR, help='training data directory')
    parser.add_argument('--ling_dir', type=str, default=LING_DIR, help='training data directory')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='checkpoint path')


    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device("cuda" if args.gpu_use else "cpu")

    h = 256  # hidden dim (channels)
    k = 3  # kernel size
    # Initialize
    train_seq2seq = True
    train_postnet = True

    h_c = hparams.converter_channels
    h_p = hparams.postnet_channels
    k = hparams.kernel_size
    s = hparams.stride
    out_dim = int(hparams.fft_size / 2 + 1)
    gru_unit = hparams.gru_unit
    f = hparams.reference_filters
    if train_seq2seq:
        styletoken = GST(in_dim=hparams.num_mels, gru_unit=gru_unit, num_gst=hparams.num_gst, style_att_dim=hparams.style_att_dim,
                         num_heads=hparams.num_head, convolutions=f).to(device)
        seq2seq = NEU2EMO(in_dim=hparams.num_ling, out_dim=hparams.num_mels, dropout=hparams.dropout, preattention=[(h_c, k, 1), (h_c, k, 3)],
            convolutions=[(h_c, k, 1), (h_c, k, 3), (h_c, k, 9), (h_c, k, 27), (h_c, k, 1)], style_embed_dim=hparams.style_att_dim).to(device)
    if train_postnet:
        postnet = MEL2LIN(in_dim=hparams.num_mels, out_dim=out_dim, style_embed_dim=hparams.style_att_dim, dropout=hparams.dropout,
            convolutions=[(h_p, k, 1), (h_p, k, 3), (2 * h_p, k, 1), (2 * h_p, k, 3)]).to(device)
    model = EVCModel(styletoken, seq2seq, postnet, mel_dim=hparams.num_mels, linear_dim=out_dim).to(device)

    checkpoint = torch.load(args.checkpoint)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    Xling_name = 'ema00001.npy'
    Xmel_name = 'ema00001.npy'
    happy_name = 'emc00101.npy'
    sad_name = 'ema00201.npy'
    angry_name = 'eme00301.npy'
    Xling = np.load(join(args.ling_dir, Xling_name))
    Xling = norm_meanvar(Xling, np.load(join(DATA_ROOT, 'feat', 'stat_linguistic_frame.npy')))

    melX = np.load(join(args.mel_dir, Xmel_name))
    happy = np.load(join(args.mel_dir, happy_name))
    sad = np.load(join(args.mel_dir, sad_name))
    angry = np.load(join(args.mel_dir, angry_name))

    lingX = torch.from_numpy(Xling).unsqueeze(0).to(device)
    melX = torch.from_numpy(melX).unsqueeze(0).to(device)
    happy = torch.from_numpy(happy).unsqueeze(0).to(device)
    sad = torch.from_numpy(sad).unsqueeze(0).to(device)
    angry = torch.from_numpy(angry).unsqueeze(0).to(device)

    style_n, mel_output, linear_output = model(lingX, melX)
    style_h, happy_mel_output, happy_linear_output = model(lingX, happy)
    style_s, sad_mel_output, sad_linear_output = model(lingX, sad)
    style_a, angry_mel_output, angry_linear_output = model(lingX, angry)

    linear_output = linear_output[0].data.cpu().numpy()
    signal = audio.inv_spectrogram(linear_output.T)
    signal /= np.max(np.abs(signal))
    path = join(args.result_dir, Xmel_name.replace('.npy', '.wav'))
    audio.save_wav(signal, path)

    happy_linear_output = happy_linear_output[0].data.cpu().numpy()
    signal = audio.inv_spectrogram(happy_linear_output.T)
    signal /= np.max(np.abs(signal))
    path = join(args.result_dir, happy_name.replace('.npy', '.wav'))
    audio.save_wav(signal, path)

    sad_linear_output = sad_linear_output[0].data.cpu().numpy()
    signal = audio.inv_spectrogram(sad_linear_output.T)
    signal /= np.max(np.abs(signal))
    path = join(args.result_dir, sad_name.replace('.npy', '.wav'))
    audio.save_wav(signal, path)

    angry_linear_output = angry_linear_output[0].data.cpu().numpy()
    signal = audio.inv_spectrogram(angry_linear_output.T)
    signal /= np.max(np.abs(signal))
    path = join(args.result_dir, angry_name.replace('.npy', '.wav'))
    audio.save_wav(signal, path)

    #mel_output = mel_output[0].data.cpu().numpy()
    #path = join(args.result_dir, l)
    #np.save(path, mel_output)

    print('%s' % melX)


