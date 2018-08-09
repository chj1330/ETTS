import tensorflow as tf
# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Audio:

    num_mels=80,
    fmin=125,
    fmax=11025,
    fft_size=1024,
    hop_size=256,
    sample_rate=22050,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    # whether to rescale waveform or not.
    # Let x is an input waveform, rescaled waveform y is given by:
    # y = x / np.abs(x).max() * rescaling_max
    rescaling=False,
    rescaling_max=0.999,
    # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db, causing clipping noise.
    # If False, assertion is added to ensure no clipping happens.
    allow_clipping_in_normalization=True,

    n_speakers=6,
    speaker_embed_dim=16,
    num_ling=342,
    speaker_embedding = False,
    style_embedding = True,
    # Maximum number of input text length
    # try setting larger value if you want to give very long text input
    dropout=1 - 0.95,
    kernel_size=3,

    converter_channels=256,
    # Note: large converter channels requires significant computational cost
    postnet_channels=256,
    gru_unit=256, #reference_depth
    reference_filters = [32, 32, 64, 64, 128, 128],
    num_gst = 10,
    num_head = 4,
    stride = 2,
    style_att_dim = 256,
    style_embed_depth = 256,
    # Data loader
    pin_memory=True,
    num_workers=8,  # Set it to 1 when in Windows (MemoryError, THAllocator.c 0x5)

    # Loss
    masked_loss_weight=0.5,  # (1-w)*loss + w * masked_loss
    binary_divergence_weight=0.1,  # set 0 to disable

    # Training:
    batch_size=12,
    adam_beta1=0.5,
    adam_beta2=0.9,
    adam_eps=1e-6,
    amsgrad=False,
    initial_learning_rate=0.01,  # 0.01,
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=10000,
    weight_decay=0.0,

    # Save
    checkpoint_interval=50,
    eval_interval=50,

    power=1.4,  # Power to raise magnitudes to prior to phase retrieval

    # GC:
    # Forced garbage collection probability
    # Use only when MemoryError continues in Windows (Disabled by default)
    #gc_probability = 0.001,

    # json_meta mode only
    # 0: "use all",
    # 1: "ignore only unmatched_alignment",
    # 2: "fully ignore recognition",
    ignore_recognition_level=2,

)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
