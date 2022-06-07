# load packages
import random, os, time
import numpy as np
import torch
import librosa
import soundfile as sf

# from utils import compute_style, load_F0, load_starganv2, load_vocoder, preprocess, speakers
from utils import preprocess, speakers, F0_model, starganv2, vocoder, compute_style

def convert(type, speaker, speakers, F0_model, vocoder, starganv2):
    '''@lw
    random sample 10 files and transform them to different speakers
    :type: style or mapping
    :speaker: index of the speaker or the speaker name
    '''

    # @lw: check the style
    assert type in [
        'style', 'sty', 'mapping', 'map'
    ], 'we only support two conversion manner: style and mapping.'
    if type in ['style', 'sty']:
        type = 'sty'
    else:
        type = 'map'

    # @lw: unify the speaker to the speaker name
    if isinstance(speaker, int):
        for k, v in speakers.items():
            if v == speaker:
                speaker = k
        assert isinstance(speaker, int), 'we only support the following speaker indexes: {}.'.format('; '.join(
            speakers.values()))
    else:
        # @lw: check whether the speaker in the list
        assert speaker in speakers.keys(
        ), 'we only support the following speaker name: {}.'.format('; '.join(
            speakers.keys()))

    # @lw: config the path
    pred_path = os.path.join(os.getcwd(), 'Pred/yisa')
    out_path = os.path.join(os.getcwd(), 'Output')

    # @lw: set reference
    speaker_dicts = {}
    for k, v in speakers.items():
        # all contain the speaker voice
        speaker_path = os.path.join(pred_path, k)
        ref_path = os.path.join(speaker_path, k + '_00017.wav')
        if type == 'sty':  # only style use reference
            speaker_dicts[k] = (ref_path, v)
        else:
            speaker_dicts[k] = ('', v)

    # @lw: compute reference embeddings
    reference_embeddings = compute_style(speaker_dicts)

    # @lw: select 10 audios to convert
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
            file_name = '{sp}_to_{cvt}_{sty}_{sf}'.format(sp=speaker,
                                                          sty=type,
                                                          cvt=key,
                                                          sf=suffix)
            converted_speaker = os.path.join(out_path, key)
            file_path = os.path.join(converted_speaker, file_name)
            sf.write(file_path, samplerate=24000, data=wave)

    end = time.time()
    print('{} total processing time: {:.3f} sec'.format(type, end - start))


if __name__ == "__main__":

    # F0_model = load_F0()
    # vocoder = load_vocoder()
    # starganv2 = load_starganv2()

    for speaker, idx in speakers.items():
        print(speaker)
        convert('sty', speaker, speakers, F0_model, vocoder, starganv2)
        convert('map', speaker, speakers, F0_model, vocoder, starganv2)