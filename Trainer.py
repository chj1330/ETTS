import torch
from torch import nn
from os.path import join
from tqdm import tqdm
from lrschedule import noam_learning_rate_decay
from util import logit, masked_mean, sequence_mask, prepare_spec_image
import audio
import numpy as np
from warnings import warn

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=False)

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
            raise RuntimeError("Mask is None")

        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask_.sum()

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, writer, checkpoint_dir, device, hparams):
        self.model = model
        self.hparams = hparams
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.optimizer = optimizer
        self.checkpoint_interval = hparams.checkpoint_interval
        self.eval_interval = hparams.eval_interval
        self.w = hparams.binary_divergence_weight
        self.epoch = hparams.nepochs
        self.fs = hparams.sample_rate

    def spec_loss(self, y_hat, y, mask, priority_bin=None, priority_w=0):
        masked_l1 = MaskedL1Loss()
        l1 = nn.L1Loss()

        w = self.hparams.masked_loss_weight

        # L1 loss
        if w > 0:
            assert mask is not None
            l1_loss = w * masked_l1(y_hat, y, mask=mask) + (1 - w) * l1(y_hat, y)
        else:
            assert mask is None
            l1_loss = l1(y_hat, y)

        # Priority L1 loss
        if priority_bin is not None and priority_w > 0:
            if w > 0:
                priority_loss = w * masked_l1(
                    y_hat[:, :, :priority_bin], y[:, :, :priority_bin], mask=mask) \
                                + (1 - w) * l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
            else:
                priority_loss = l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
            l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

        # Binary divergence loss
        if self.w <= 0:
            binary_div = y.data.new(1).zero_()
        else:
            y_hat_logits = logit(y_hat)
            z = -y * y_hat_logits + torch.log1p(torch.exp(y_hat_logits))
            if w > 0:
                binary_div = w * masked_mean(z, mask) + (1 - w) * z.mean()
            else:
                binary_div = z.mean()

        return l1_loss, binary_div


    def train(self, train_seq2seq, train_postnet, global_epoch=1, global_step=0):
        while global_epoch < self.epoch:
            running_loss = 0.
            running_linear_loss = 0.
            running_mel_loss = 0.
            for step, (ling, mel, linear, lengths, speaker_ids) in enumerate(tqdm(self.train_loader)):
                self.model.train()
                ismultispeaker = speaker_ids is not None
                # Learn rate scheduler
                current_lr = noam_learning_rate_decay(self.hparams.initial_learning_rate, global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.optimizer.zero_grad()

                # Transform data to CUDA device
                if train_seq2seq :
                    ling = ling.to(self.device)
                    mel = mel.to(self.device)
                if train_postnet :
                    linear = linear.to(self.device)
                lengths = lengths.to(self.device)
                speaker_ids = speaker_ids.to(self.device) if ismultispeaker else None
                target_mask = sequence_mask(lengths, max_len=mel.size(1)).unsqueeze(-1)

                # Apply model
                if train_seq2seq and train_postnet:
                    _, mel_outputs, linear_outputs = self.model(ling, mel, speaker_ids=speaker_ids)
                #elif train_seq2seq:
                #    mel_style = self.model.gst(tmel)
                #    style_embed = mel_style.expand_as(smel)
                #    mel_input = smel + style_embed
                #    mel_outputs = self.model.seq2seq(mel_input)
                #    linear_outputs = None
                #elif train_postnet:
                #    linear_outputs = self.model.postnet(smel)
                #    mel_outputs = None

                # Losses
                if train_seq2seq:
                    mel_l1_loss, mel_binary_div = self.spec_loss(mel_outputs, mel, target_mask)
                    mel_loss = (1 - self.w) * mel_l1_loss + self.w * mel_binary_div
                if train_postnet:
                    linear_l1_loss, linear_binary_div = self.spec_loss(linear_outputs, linear, target_mask)
                    linear_loss = (1 - self.w) * linear_l1_loss + self.w * linear_binary_div

                # Combine losses
                if train_seq2seq and train_postnet:
                    loss = mel_loss + linear_loss
                elif train_seq2seq:
                    loss = mel_loss
                elif train_postnet:
                    loss = linear_loss

                # Update
                loss.backward()
                self.optimizer.step()
                # Logs
                if train_seq2seq:
                    self.writer.add_scalar("mel loss", float(mel_loss.item()), global_step)
                    self.writer.add_scalar("mel_l1_loss", float(mel_l1_loss.item()), global_step)
                    self.writer.add_scalar("mel_binary_div_loss", float(mel_binary_div.item()), global_step)
                if train_postnet:
                    self.writer.add_scalar("linear_loss", float(linear_loss.item()), global_step)
                    self.writer.add_scalar("linear_l1_loss", float(linear_l1_loss.item()), global_step)
                    self.writer.add_scalar("linear_binary_div_loss", float(
                        linear_binary_div.item()), global_step)
                self.writer.add_scalar("loss", float(loss.item()), global_step)
                self.writer.add_scalar("learning rate", current_lr, global_step)

                global_step += 1
                running_loss += loss.item()
                running_linear_loss += linear_loss.item()
                running_mel_loss += mel_loss.item()

            if (global_epoch % self.checkpoint_interval == 0):
                self.save_checkpoint(global_step, global_epoch)
            if global_epoch % self.eval_interval == 0:
                self.save_states(global_epoch, mel_outputs, linear_outputs, ling, mel, linear, lengths)
            self.eval_model(global_epoch, train_seq2seq, train_postnet)
            avg_loss = running_loss / len(self.train_loader)
            avg_linear_loss = running_linear_loss / len(self.train_loader)
            avg_mel_loss = running_mel_loss / len(self.train_loader)
            self.writer.add_scalar("train loss (per epoch)", avg_loss, global_epoch)
            self.writer.add_scalar("train linear loss (per epoch)", avg_linear_loss, global_epoch)
            self.writer.add_scalar("train mel loss (per epoch)", avg_mel_loss, global_epoch)
            print("Train Loss: {}".format(avg_loss))
            global_epoch += 1


    def eval_model(self, global_epoch, train_seq2seq, train_postnet):
        happy_ref = np.load('../feat/Acoustic_frame/mel/emc00103.npy')
        happy_ref = torch.from_numpy(happy_ref).unsqueeze(0)
        sad_ref = np.load('../feat/Acoustic_frame/mel/ema00203.npy')
        sad_ref = torch.from_numpy(sad_ref).unsqueeze(0)
        angry_ref = np.load('../feat/Acoustic_frame/mel/eme00303.npy')
        angry_ref = torch.from_numpy(angry_ref).unsqueeze(0)
        running_loss = 0.
        running_linear_loss = 0.
        running_mel_loss = 0.
        for step, (ling, mel, linear, lengths, speaker_ids) in enumerate(self.valid_loader):
            self.model.eval()
            ismultispeaker = speaker_ids is not None
            if train_seq2seq:
                ling = ling.to(self.device)
                mel = mel.to(self.device)
                happy_ref = happy_ref.to(self.device)
                sad_ref = sad_ref.to(self.device)
                angry_ref = angry_ref.to(self.device)
            if train_postnet:
                linear = linear.to(self.device)
            lengths = lengths.to(self.device)
            speaker_ids = speaker_ids.to(self.device) if ismultispeaker else None
            target_mask = sequence_mask(lengths, max_len=mel.size(1)).unsqueeze(-1)
            with torch.no_grad():
            # Apply model
                if train_seq2seq and train_postnet:
                    _, mel_outputs, linear_outputs = self.model(ling, mel, speaker_ids=speaker_ids)

                """
                elif train_seq2seq:
                    mel_style = self.model.gst(tmel)
                    style_embed = mel_style.expand_as(smel)
                    mel_input = smel + style_embed
                    mel_outputs = self.model.seq2seq(mel_input)
                    linear_outputs = None
                elif train_postnet:
                    linear_outputs = self.model.postnet(tmel)
                    mel_outputs = None
                """

            # Losses
            if train_seq2seq:
                mel_l1_loss, mel_binary_div = self.spec_loss(mel_outputs, mel, target_mask)
                mel_loss = (1 - self.w) * mel_l1_loss + self.w * mel_binary_div
            if train_postnet:
                linear_l1_loss, linear_binary_div = self.spec_loss(linear_outputs, linear, target_mask)
                linear_loss = (1 - self.w) * linear_l1_loss + self.w * linear_binary_div

            # Combine losses
            if train_seq2seq and train_postnet:
                loss = mel_loss + linear_loss
            elif train_seq2seq:
                loss = mel_loss
            elif train_postnet:
                loss = linear_loss
            running_loss += loss.item()
            running_linear_loss += linear_loss.item()
            running_mel_loss += mel_loss.item()
        B = ling.size(0)
        if ismultispeaker :
            speaker_ids = np.zeros(B)
            speaker_ids = torch.LongTensor(speaker_ids).to(self.device)
        else :
            speaker_ids = None
        _, happy_mel_outputs, happy_linear_outputs = self.model(ling, happy_ref, speaker_ids)
        _, sad_mel_outputs, sad_linear_outputs = self.model(ling, sad_ref, speaker_ids)
        _, angry_mel_outputs, angry_linear_outputs = self.model(ling, angry_ref, speaker_ids)

        if global_epoch % self.eval_interval == 0:
            for idx in range(B):
                if mel_outputs is not None:
                    happy_mel_output = happy_mel_outputs[idx].cpu().data.numpy()
                    happy_mel_output = prepare_spec_image(audio._denormalize(happy_mel_output))
                    self.writer.add_image("(Eval) Happy mel spectrogram {}".format(idx), happy_mel_output, global_epoch)

                    sad_mel_output = sad_mel_outputs[idx].cpu().data.numpy()
                    sad_mel_output = prepare_spec_image(audio._denormalize(sad_mel_output))
                    self.writer.add_image("(Eval) Sad mel spectrogram {}".format(idx), sad_mel_output, global_epoch)

                    angry_mel_output = angry_mel_outputs[idx].cpu().data.numpy()
                    angry_mel_output = prepare_spec_image(audio._denormalize(angry_mel_output))
                    self.writer.add_image("(Eval) Angry mel spectrogram {}".format(idx), angry_mel_output, global_epoch)

                    mel_output = mel_outputs[idx].cpu().data.numpy()
                    mel_output = prepare_spec_image(audio._denormalize(mel_output))
                    self.writer.add_image("(Eval) Predicted mel spectrogram {}".format(idx), mel_output, global_epoch)

                    mel1 = mel[idx].cpu().data.numpy()
                    mel1 = prepare_spec_image(audio._denormalize(mel1))
                    self.writer.add_image("(Eval) Source mel spectrogram {}".format(idx), mel1, global_epoch)

                if linear_outputs is not None:
                    linear_output = linear_outputs[idx].cpu().data.numpy()
                    spectrogram = prepare_spec_image(audio._denormalize(linear_output))
                    self.writer.add_image("(Eval) Predicted spectrogram {}".format(idx), spectrogram, global_epoch)
                    signal = audio.inv_spectrogram(linear_output.T)
                    signal /= np.max(np.abs(signal))
                    path = join(self.checkpoint_dir, "epoch{:09d}_{}_predicted.wav".format(global_epoch, idx))
                    audio.save_wav(signal, path)
                    try:
                        self.writer.add_audio("(Eval) Predicted audio signal {}".format(idx), signal, global_epoch, sample_rate=self.fs)
                    except Exception as e:
                        warn(str(e))
                        pass

                    happy_linear_output = happy_linear_outputs[idx].cpu().data.numpy()
                    spectrogram = prepare_spec_image(audio._denormalize(happy_linear_output))
                    self.writer.add_image("(Eval) Happy spectrogram {}".format(idx), spectrogram, global_epoch)
                    signal = audio.inv_spectrogram(happy_linear_output.T)
                    signal /= np.max(np.abs(signal))
                    path = join(self.checkpoint_dir, "epoch{:09d}_{}_happy.wav".format(global_epoch, idx))
                    audio.save_wav(signal, path)
                    try:
                        self.writer.add_audio("(Eval) Happy audio signal {}".format(idx), signal, global_epoch, sample_rate=self.fs)
                    except Exception as e:
                        warn(str(e))
                        pass

                    angry_linear_output = angry_linear_outputs[idx].cpu().data.numpy()
                    spectrogram = prepare_spec_image(audio._denormalize(angry_linear_output))
                    self.writer.add_image("(Eval) Angry spectrogram {}".format(idx), spectrogram, global_epoch)
                    signal = audio.inv_spectrogram(angry_linear_output.T)
                    signal /= np.max(np.abs(signal))
                    path = join(self.checkpoint_dir, "epoch{:09d}_{}_angry.wav".format(global_epoch, idx))
                    audio.save_wav(signal, path)
                    try:
                        self.writer.add_audio("(Eval) Angry audio signal {}".format(idx), signal, global_epoch, sample_rate=self.fs)
                    except Exception as e:
                        warn(str(e))
                        pass

                    sad_linear_output = sad_linear_outputs[idx].cpu().data.numpy()
                    spectrogram = prepare_spec_image(audio._denormalize(sad_linear_output))
                    self.writer.add_image("(Eval) Sad spectrogram {}".format(idx), spectrogram, global_epoch)
                    signal = audio.inv_spectrogram(sad_linear_output.T)
                    signal /= np.max(np.abs(signal))
                    path = join(self.checkpoint_dir, "epoch{:09d}_{}_sad.wav".format(global_epoch, idx))
                    audio.save_wav(signal, path)
                    try:
                        self.writer.add_audio("(Eval) Sad audio signal {}".format(idx), signal, global_epoch, sample_rate=self.fs)
                    except Exception as e:
                        warn(str(e))
                        pass


                    linear1 = linear[idx].cpu().data.numpy()
                    spectrogram = prepare_spec_image(audio._denormalize(linear1))
                    self.writer.add_image("(Eval) Target spectrogram {}".format(idx), spectrogram, global_epoch)
                    signal = audio.inv_spectrogram(linear1.T)
                    signal /= np.max(np.abs(signal))
                    try:
                        self.writer.add_audio("(Eval) Target audio signal {}".format(idx), signal, global_epoch, sample_rate=self.fs)
                    except Exception as e:
                        warn(str(e))
                        pass

        avg_loss = running_loss / len(self.valid_loader)
        avg_linear_loss = running_linear_loss / len(self.valid_loader)
        avg_mel_loss = running_mel_loss / len(self.valid_loader)
        self.writer.add_scalar("valid loss (per epoch)", avg_loss, global_epoch)
        self.writer.add_scalar("valid linear loss (per epoch)", avg_linear_loss, global_epoch)
        self.writer.add_scalar("valid mel loss (per epoch)", avg_mel_loss, global_epoch)
        print("Valid Loss: {}".format(avg_loss))




    def save_states(self, global_epoch, mel_outputs, linear_outputs, ling, mel, linear, lengths):
        print("Save intermediate states at epoch {}".format(global_epoch))

        # idx = np.random.randint(0, len(input_lengths))
        idx = min(1, len(lengths) - 1)

        # Predicted mel spectrogram
        if mel_outputs is not None:
            mel_output = mel_outputs[idx].cpu().data.numpy()
            mel_output = prepare_spec_image(audio._denormalize(mel_output))
            self.writer.add_image("Predicted mel spectrogram", mel_output, global_epoch)
        # Predicted spectrogram
        if linear_outputs is not None:
            linear_output = linear_outputs[idx].cpu().data.numpy()
            spectrogram = prepare_spec_image(audio._denormalize(linear_output))
            self.writer.add_image("Predicted spectrogram", spectrogram, global_epoch)
            # Predicted audio signal
            signal = audio.inv_spectrogram(linear_output.T)
            signal /= np.max(np.abs(signal))
            path = join(self.checkpoint_dir, "epoch{:09d}_predicted.wav".format(global_epoch))
            try:
                self.writer.add_audio("Predicted audio signal", signal, global_epoch, sample_rate=self.fs)
            except Exception as e:
                warn(str(e))
                pass
            audio.save_wav(signal, path)

        # Target mel spectrogram

        if mel_outputs is not None:
            #ling = ling[idx].cpu().data.numpy()
            #mel = prepare_spec_image(audio._denormalize(mel))
            #self.writer.add_image("Source mel spectrogram", ling, global_epoch)
            mel = mel[idx].cpu().data.numpy()
            mel = prepare_spec_image(audio._denormalize(mel))
            self.writer.add_image("Target mel spectrogram", mel, global_epoch)
        if linear_outputs is not None:
            linear = linear[idx].cpu().data.numpy()
            spectrogram = prepare_spec_image(audio._denormalize(linear))
            self.writer.add_image("Target spectrogram", spectrogram, global_epoch)
            # Target audio signal
            signal = audio.inv_spectrogram(linear.T)
            signal /= np.max(np.abs(signal))
            try:
                self.writer.add_audio("Target audio signal", signal, global_epoch, sample_rate=self.fs)
            except Exception as e:
                warn(str(e))
                pass



    def save_checkpoint(self, global_step, global_epoch):
        checkpoint_path = join(self.checkpoint_dir, "checkpoint_epoch{:09d}.pth".format(global_epoch))
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": global_step,
            "global_epoch": global_epoch,
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)






