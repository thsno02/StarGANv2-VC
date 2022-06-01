# load packages
import random, os, time
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder

from parallel_wavegan.utils import load_model  # load vocoder

# %matplotlib inline


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
            label = torch.LongTensor([speaker]).to('cuda')
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(
                torch.randn(1, latent_dim).to('cuda'), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave).to('cuda')

            with torch.no_grad():
                label = torch.LongTensor([speaker])
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


def load_starganv2(gan_path='Models/yisa/epoch_00150.pth'):
    '''@lw
    return starGANv2
    :gan_path: default = Models/yisa/epoch_00150.pth'
    '''

    assert torch.cuda.is_available(), "CUDA is unavailable."

    with open('Models/yisa/config.yml') as f:
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


def style_convert(speaker, speakers, F0_model, vocoder, starganv2):
    '''@lw
    random sample 10 files and transform them to different speakers
    '''

    if isinstance(speaker, int):
        speaker = speakers[speaker]

    pred_path = os.path.join(os.getcwd(), 'Pred/yisa')

    # with reference, using style encoder
    speaker_dicts = {}
    for k, v in speakers.items():
        # all contain the speaker voice
        ref_path = os.path.join(pred_path, v + '_0017.wav')
        speaker_dicts[v] = (ref_path, k)

    reference_embeddings = compute_style(speaker_dicts)

    speaker_path = os.path.join(pred_path, speaker)
    audio_files = os.listdir(speaker_path)
    # filter .DS_Store
    audio_files = [i for i in audio_files if i.endswith('.wav')]
    random.seed(2022)
    selected_audio = random.sample(audio_files, 10)
    start = time.time()

    for file_name in selected_audio:
        wav_path = os.path.join(speaker_path, file_name)
        audio, source_sr = librosa.load(wav_path, sr=24000)
        audio = audio / np.max(np.abs(audio))
        audio.dtype = np.float32

        # conversion
        source = preprocess(audio).to('cuda:0')
        keys = []
        converted_samples = {}
        reconstructed_samples = {}
        converted_mels = {}

        for key, (ref, _) in reference_embeddings.items():
            with torch.no_grad():
                f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
                out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

                c = out.transpose(-1, -2).squeeze().to('cuda')
                y_out = vocoder.inference(c)
                y_out = y_out.view(-1).cpu()

                if key not in speaker_dicts or speaker_dicts[key][0] == "":
                    recon = None
                else:
                    wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                    mel = preprocess(wave)
                    c = mel.transpose(-1, -2).squeeze().to('cuda')
                    recon = vocoder.inference(c)
                    recon = recon.view(-1).cpu().numpy()

            converted_samples[key] = y_out.numpy()
            reconstructed_samples[key] = recon

            converted_mels[key] = out

            keys.append(key)

        for key, wave in converted_samples.items():
            suffix = file_name.split('_')[-1]
            file_name = '{sp}_to_{cvt}_style_{sf}'.format(sp=speaker,
                                                          cvt=key,
                                                          sf=suffix)
            sf.write(file_name, samplerate=24000, data=wave)

    end = time.time()
    print('total processing time: %.3f sec' % (end - start))


def mapping_convert(speaker, speakers, F0_model, vocoder, starganv2):
    '''@lw
    random sample 10 files and transform them to different speakers
    '''
    if isinstance(speaker, int):
        speaker = speakers[speaker]

    pred_path = os.path.join(os.getcwd(), 'Pred/yisa')

    # with reference, using style encoder
    speaker_dicts = {}
    for k, v in speakers.items():
        speaker_dicts[v] = ('', k)

    reference_embeddings = compute_style(speaker_dicts)

    speaker_path = os.path.join(pred_path, speaker)
    audio_files = os.listdir(speaker_path)
    # filter .DS_Store
    audio_files = [i for i in audio_files if i.endswith('.wav')]
    random.seed(2022)
    selected_audio = random.sample(audio_files, 10)
    start = time.time()

    for file_name in selected_audio:
        wav_path = os.path.join(speaker_path, file_name)
        audio, source_sr = librosa.load(wav_path, sr=24000)
        audio = audio / np.max(np.abs(audio))
        audio.dtype = np.float32

        # conversion
        source = preprocess(audio).to('cuda:0')
        keys = []
        converted_samples = {}
        reconstructed_samples = {}
        converted_mels = {}

        for key, (ref, _) in reference_embeddings.items():
            with torch.no_grad():
                f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
                out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

                c = out.transpose(-1, -2).squeeze().to('cuda')
                y_out = vocoder.inference(c)
                y_out = y_out.view(-1).cpu()

                if key not in speaker_dicts or speaker_dicts[key][0] == "":
                    recon = None
                else:
                    wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                    mel = preprocess(wave)
                    c = mel.transpose(-1, -2).squeeze().to('cuda')
                    recon = vocoder.inference(c)
                    recon = recon.view(-1).cpu().numpy()

            converted_samples[key] = y_out.numpy()
            reconstructed_samples[key] = recon

            converted_mels[key] = out

            keys.append(key)

        for key, wave in converted_samples.items():
            suffix = file_name.split('_')[-1]
            file_name = '{sp}_to_{cvt}_mapping_{sf}'.format(sp=speaker,
                                                            cvt=key,
                                                            sf=suffix)
            sf.write(file_name, samplerate=24000, data=wave)

    end = time.time()
    print('total processing time: %.3f sec' % (end - start))


if __name__ == "__main__":

    F0_model = load_F0()
    vocoder = load_vocoder()
    starganv2 = load_starganv2()

    speakers = {
        0: 'Li_Fanping',
        1: 'Shi_Zhuguo',
        2: 'Wang_Cheng',
        3: 'Wang_Kun',
        4: 'Zhao_Lijian',
        5: 'Hua_Chunying',
        6: 'Luo_Xiang',
        7: 'Li_Gan',
        8: 'Dong_Mingzhu',
        9: 'Ma_Yun'
    }

    for k, speaker in speakers.items():
        style_convert(speaker, speakers, F0_model, vocoder, starganv2)
        mapping_convert(speaker, speakers, F0_model, vocoder, starganv2)