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

    start = time.time()

    # conversion
    source = preprocess(audio).to('cuda:0')
    converted_audio = None

    for key, (ref, _) in reference_embeddings.items():
        with torch.no_grad():
            f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
            out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

            c = out.transpose(-1, -2).squeeze().to('cuda')
            y_out = vocoder.inference(c)
            y_out = y_out.view(-1).cpu()

        converted_audio = y_out.numpy()

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
sd.default.device = input_device, output_device
fs = 24000
sd.default.samplerate = fs # set sample rate
sd.default.channels = 1, 2 # one input channel, two output channel

# set speak
speaker = 'Shi_Zhuguo'


# TODO: how to set the duration?
duration = 15 # seconds


if __name__ == "__main__":
    print('begin recording')
    # record voice
    audio = sd.rec(int(duration * fs), dtype = 'float32')
    # pre-process audio
    audio = audio / np.max(np.abs(audio))
    # convert audio to target speaker toned
    converted_audio = convert(audio, speaker, F0_model, vocoder, starganv2)
    sd.playrec(converted_audio, fs)