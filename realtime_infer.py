#!/usr/bin/env python3
"""Pass input directly to output.

See https://www.assembla.com/spaces/portaudio/subversion/source/HEAD/portaudio/trunk/test/patest_wire.c

"""
import argparse
import logging
import os.path
import pickle

from vocoder import *
from models import *
MAX_WAV_VALUE = 32768.0



parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--input-device", type=int, help="input device ID")
parser.add_argument("-o", "--output-device", type=int, help="output device ID")
parser.add_argument("-c", "--channels", type=int, default=2,
                    help="number of channels")
parser.add_argument("-t", "--dtype", help="audio data type")
parser.add_argument("-s", "--samplerate", type=float, help="sampling rate", default=24000)
parser.add_argument("-b", "--blocksize", type=int, help="block size", default=32 * 300)
parser.add_argument("-l", "--latency", type=float, help="latency in seconds", default=0)
parser.add_argument("-p", "--speaker", type=float, help="spekaer id", default=8)
parser.add_argument("-m", "--mel-buffer-size", type=float, help="mel buffer size", default=192)
parser.add_argument("-w", "--wave-buffer-size", type=float, help="wave buffer size in frame", default=10)

args = parser.parse_args()


generator = load_vocoder()
starganv2, F0_model = load_models()
ref = compute_style('reference.wav', starganv2, args.speaker)

def next_waves(wave, wave_next, wave_average=None):
    wave_left = wave_next.shape[-1] - args.wave_buffer_size * 300 - args.blocksize
    wave_right = wave_next.shape[-1] - args.wave_buffer_size * 300

    a = wave.squeeze()[-300:]
    b = wave_next[..., wave_left - a.shape[-1]:wave_right].squeeze()
    print(a.shape, b.shape)
    b = b[:a.shape[-1] + 300]

    wave = wave_next[..., wave_left - a.shape[-1]:wave_right + 1 + 300].squeeze()
    wave = np.clip(wave, -1, 1)
    wave = wave[a.shape[-1] + 1:]
    wave_return = wave[:-300]
    wave_buffer_return = wave[-300:]

    if wave_average is not None:
        buffer_weight = np.linspace(1, 0, 300)
        wave_return[0:300] = buffer_weight * wave_average + (1 - buffer_weight) * wave_return[0:300]

    return wave_return, wave_buffer_return


try:
    import sounddevice as sd

    mel_buffer = torch.zeros(1, 80, int(args.mel_buffer_size - args.blocksize / 300)).to('cuda')
    callback_status = sd.CallbackFlags()

    wave_buffer = np.zeros(args.mel_buffer_size * 300)
    previous_wave = None
    wave_average = None
    noisy_part = None

    def callback(indata, outdata, frames, time, status):
        global callback_status
        global mel_buffer
        global wave_buffer
        global previous_wave
        global wave_average

        callback_status |= status
        with torch.no_grad():
            audio = indata[:, 0].squeeze()

            # wave buffer
            wave_buffer = np.concatenate((wave_buffer, audio), axis=-1)
            buffer_size = args.mel_buffer_size * 300 - args.blocksize
            buffer_cut = int(wave_buffer.shape[-1] - buffer_size)
            wave_buffer = wave_buffer[..., max(0, buffer_cut):]
            print('mel_buffer', wave_buffer.shape)

            mel = preprocess_GPU(wave_buffer)
            mel = mel[..., 1:]
            print('mel', mel.shape)


            out = convert(mel, starganv2, F0_model, ref, args.speaker)
            mel_left = out.shape[-1] - args.wave_buffer_size  - int(args.blocksize / 300)
            mel_right = out.shape[-1] - args.wave_buffer_size
            mel_out = out[..., mel_left:mel_right]

            # out = mel
            wave = generator(out.squeeze().unsqueeze(0))
            wave = wave.cpu().numpy()
            wave.dtype = np.float32


            if previous_wave is None:
                wave_left = wave.shape[-1] - args.wave_buffer_size * 300 - args.blocksize
                wave_right = wave.shape[-1] - args.wave_buffer_size * 300
                print('wave range', wave_left, wave_right)
                print('wave shape', wave.shape)
                wave = wave[..., wave_left:wave_right]
                previous_wave = wave
            else:
                wave, wave_average = next_waves(previous_wave, wave, wave_average)
                previous_wave = wave


            wave = np.expand_dims(wave.squeeze(), axis=1)

        out = np.repeat(wave, 2, axis=1).squeeze()
        if out.shape[0] == 2:
            outdata[:] = out.transpose()
        else:
            outdata[:] = out


    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print("#" * 80)
        print("press Return to quit")
        print("#" * 80)
        input()

    if callback_status:
        logging.warning(str(callback_status))
except BaseException as e:
    # This avoids printing the traceback, especially if Ctrl-C is used.
    raise SystemExit(str(e))