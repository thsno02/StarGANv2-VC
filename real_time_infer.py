import sounddevice as sd
import torch
from utils import compute_style, F0_model, starganv2, vocoder, preprocess, speakers
import numpy as np
import time


def convert(audio, speaker, F0_model, vocoder, starganv2):
    '''@lw
    :speaker: the speaker name
    '''

    # # @lw: unify the speaker to the speaker name
    # if isinstance(speaker, int):
    #     speaker = speakers[speaker]
    # else:
    #     # @lw: check whether the speaker in the list
    #     assert speaker in speakers.values(
    #     ), 'we only support the following speakers: {}.'.format('; '.join(
    #         speakers.values()))

    # @lw: set reference, get the speaker index
    speaker_dicts = {speaker: ('', speakers[speaker])}

    # @lw: compute reference embeddings
    reference_embeddings = compute_style(speaker_dicts)
    # print(f'ref embedding is {reference_embeddings}')

    start = time.time()

    # conversion
    source = preprocess(audio).to('cuda:0')
    print('length of source is {}'.format(source.size()))
    converted_audio = None

    for key, (ref, _) in reference_embeddings.items():
        with torch.no_grad():
            f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
            out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

            c = out.transpose(-1, -2).squeeze().to('cuda')
            print('c size is {}'.format(c.size()))
            y_out = vocoder.inference(c)
            y_out = y_out.view(-1).cpu()

        converted_audio = y_out.numpy()
        print('converted_audio {}'.format(len(converted_audio)))

    end = time.time()
    print('{} total processing time: {:.3f} sec'.format(type, end - start))

    return converted_audio

# enable to detect bluetooth devices, if and only if the devices are paired
sd._terminate()
sd._initialize()

# show the available devices
device_list = sd.query_devices()
print(f'the device list is: \n{device_list}.\n')
for device in device_list:
    # TODO: automatically find the available bluetooth devices
    # As for now, use headphone instead
    if 'External Microphone' in device['name']:
        input_device = device['name']
        print(f"Input device name is '{input_device}'.")
    elif 'External Headphones' in device['name']:
        output_device = device['name']
        print(f"Output device name is '{output_device}'.")

# set input and output devices
sd.default.device = 3, 5
fs = 24000
sd.default.samplerate = fs # set sample rate
sd.default.channels = 1, 1 # one input channel, two output channel

# set speak
speaker = 6
# speakers = {
#     0: 'Dong_Mingzhu',
#     1: 'Hua_Chunying',
#     2: 'Li_Fanping',
#     3: 'Li_Gan',
#     4: 'Luo_Xiang',
#     5: 'Ma_Yun',
#     6: 'Shi_Zhuguo',
#     7: 'Wang_Cheng',
#     8: 'Wang_Kun',
#     9: 'Zhao_Lijian'
# }


# TODO: how to set the duration?
duration = 5 # seconds


if __name__ == "__main__":
    print('begin recording')
    # record voice
    start_time = time.time()
    audio = sd.rec(int(duration * fs), dtype = 'float32')
    print('raw audio length is {}'.format(len(audio)))
    sd.wait() # wait to recording
    end_time = time.time()
    print('rec costs {:.4} s'.format(end_time - start_time))
    # pre-process audio
    # audio = audio / np.max(np.abs(audio))
    audio = audio.flatten() # flatten the 2D numpy array
    print('audio length is {}'.format(len(audio)))
    # convert audio to target speaker toned
    print('begin converting')
    start_time = time.time()
    converted_audio = convert(audio, speaker, F0_model, vocoder, starganv2)
    print('converted_audio length is {}'.format(len(converted_audio)))
    end_time = time.time()
    print('VC costs {:.4} s'.format(end_time - start_time))
    print('begin playing')
    start_time = time.time()
    sd.play(converted_audio, fs)
    sd.wait() # wait to playing
    end_time = time.time()
    print('play costs {:.4} s'.format(end_time - start_time))