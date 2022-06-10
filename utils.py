import yaml, os
from munch import Munch
import torch
import librosa
import torchaudio

from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder

from parallel_wavegan.utils import load_model  # load vocoder

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80,
                                              n_fft=2048,
                                              win_length=1200,
                                              hop_length=300)
mean, std = -4, 4


# def build_speakers():
#     cwd = os.getcwd()
#     data_path = os.path.join(cwd, 'Data')
#     speakers = []
#     for file in os.listdir(data_path):
#         # is directory and not raw
#         if os.path.isdir(os.path.join(data_path, file)) and '_' in file:
#             speakers.append(file)
#     speakers_dict = {}
#     for t in enumerate(speakers):
#         # @lw: key = speaker name, value = index
#         speakers_dict[t[1]] = t[0]

#     return speakers_dict


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in,
                          args.style_dim,
                          args.max_conv_dim,
                          w_hpf=args.w_hpf,
                          F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim,
                                     args.style_dim,
                                     args.num_domains,
                                     hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains,
                                 args.max_conv_dim)

    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema


def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            # @lw: speaker = idx of the speaker name
            label = torch.LongTensor([key]).to('cuda')
            latent_dim = starganv2.mapping_network.shared[0].in_features
            # @lw: get the reference embedding from the mapping network
            ref = starganv2.mapping_network(
                torch.randn(1, latent_dim).to('cuda'), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave).to('cuda')

            with torch.no_grad():
                label = torch.LongTensor([key])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)

    return reference_embeddings


def load_F0(f0_path="./Utils/JDC/bst.t7"):
    ''' @lw
    return F0 model
    :f0_path: default path is "./Utils/JDC/bst.t7"
    '''

    assert torch.cuda.is_available(), "CUDA is unavailable."
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(f0_path)['net']
    F0_model.load_state_dict(params)
    _ = F0_model.eval()
    F0_model = F0_model.to('cuda')

    return F0_model


def load_vocoder(vocoder_path="./Vocoder/checkpoint-400000steps.pkl"):
    '''@lw
    return vocoder model
    :vocoder_path: default path is "./Vocoder/checkpoint-400000steps.pkl"
    '''

    assert torch.cuda.is_available(), "CUDA is unavailable."
    vocoder = load_model(vocoder_path).to('cuda').eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()

    return vocoder


def load_starganv2(gan_path='Models/epoch_v2_00248.pth'):
    '''@lw
    return starGANv2
    :gan_path: default = Models/epoch_v2_00248.pth'
    '''

    assert torch.cuda.is_available(), "CUDA is unavailable."

    with open('Models/config.yml') as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = build_model(model_params=starganv2_config["model_params"])
    params = torch.load(gan_path, map_location='cpu')
    params = params['model_ema']
    # @lw: rebuild the parameter dictionary to elude key inconsistent issue
    for k in params:
        for s in list(params[k]):
            v = params[k][s]
            del params[k][s]
            s = '.'.join(s.split('.')[1:])
            params[k][s] = v
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')

    return starganv2


# speakers = build_speakers()
speakers = {
    0: 'Dong_Mingzhu',
    1: 'Hua_Chunying',
    2: 'Li_Fanping',
    3: 'Li_Gan',
    4: 'Luo_Xiang',
    5: 'Ma_Yun',
    6: 'Shi_Zhuguo',
    7: 'Wang_Cheng',
    8: 'Wang_Kun',
    9: 'Zhao_Lijian'
}

starganv2 = load_starganv2()
F0_model = load_F0()
vocoder = load_vocoder()

print('speakers id is {}'.format(id(speakers)))
print('starganv2 id is {}'.format(id(starganv2)))
print('F0_model id is {}'.format(id(F0_model)))
print('vocoder id is {}'.format(id(vocoder)))