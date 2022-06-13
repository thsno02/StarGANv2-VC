import sounddevice as sd
from model import Model
import numpy as np
import time
from datetime import datetime


def log(s: str):
    with open('inference.log', 'a') as f:
        f.write(s + '\n')


def get_time_dif(start_time):
    end_time = time.time()
    return round(end_time - start_time, 3)


log('------{}------'.format(datetime.now()))
model = Model()

# enable to detect bluetooth devices, if and only if the devices are paired
sd._terminate()
sd._initialize()

# show the available devices
device_list = sd.query_devices()
print(f'the device list is: \n{device_list}.\n')
for device in device_list:
    # TODO: automatically find the available bluetooth devices
    # As for now, use headphone instead
    if 'USB PnP Sound Device' in device['name']:
        input_device = device['name']
        print(f"Input device name is '{input_device}'.")
    elif 'USB PnP Sound Device' in device['name']:
        output_device = device['name']
        print(f"Output device name is '{output_device}'.")

# set input and output devices
sd.default.device = input_device, output_device
fs = 24000
sd.default.samplerate = fs  # set sample rate
sd.default.channels = 1, 2  # one input channel, two output channel

if __name__ == "__main__":
    while True:
        try:
            # set speak
            speaker = int(input('plz type the target speaker'))
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
            log('\tThe target speaker is {}. {}'.format(speaker,
                                                        model.speakers[speaker]))
            # TODO: how to set the duration?
            duration = int(input('plz type the duration (seconds)'))  # seconds
            log('\tThe duration is {} s'.format(duration))
            print('begin recording')
            # record voice
            start_time = time.time()
            audio = sd.rec(int(duration * fs), dtype='float32')
            sd.wait()  # wait to recording
            print('recording finish')
            log('\tRecording costs {} s'.format(get_time_dif(start_time)))
            # pre-process audio
            audio = audio / np.max(np.abs(audio))
            audio = audio.flatten()  # flatten the 2D numpy array
            # convert audio to target speaker tone
            print('begin converting')
            start_time = time.time()
            converted_audio = model.infer(audio, speaker)
            log('\tVC costs {:.4} s'.format(get_time_dif(start_time)))
            print('begin playing')
            start_time = time.time()
            sd.play(converted_audio, fs)
            sd.wait()  # wait to playing
            print('playing finish')
            log('\tPlaying costs {:.4} s'.format(get_time_dif(start_time)))
        except KeyboardInterrupt:
            print('Thank you.')
