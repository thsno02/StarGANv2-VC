import queue, os

import datetime
import sounddevice as sd
import soundfile as sf
import numpy as np

# enable to detect bluetooth devices, if and only if the devices are paired
sd._terminate()
sd._initialize()

# show the available devices
device_list = sd.query_devices()
print(f'the device list is: \n{device_list}.\n')
input_device = int(input('plz type the input device: '))
output_device = int(input('plz type the output device: '))
# set input and output devices
sd.default.device = input_device, output_device

fs = 24000
sd.default.samplerate = fs  # set sample rate
sd.default.channels = 1, 2  # one input channel, two output channel

while True:
        audio = np.zeros(
            shape=(24000,
                   1))  # add zero-filled buffer to promote the performance
        q = queue.Queue()

        def callback(in_data, frames, time, status):
            q.put(in_data.copy())

        try:
            with sd.InputStream(samplerate=fs,
                                device=input_device,
                                dtype='float32',
                                channels=1,
                                callback=callback):

                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    audio = np.append(audio, q.get(), axis=0)
        except KeyboardInterrupt:
            print('end recording')
        # pre-process audio
        audio = audio / np.max(np.abs(audio))
        audio = audio.flatten()  # flatten the 2D numpy array

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = input('plz type the content name: ')
        file_name = '_'.join([content, timestamp]) + '.wav'
        path = os.path.join(os.getcwd(), 'Test' + os.sep + 'Ref')
        path = os.path.join(path, file_name)
        t = sf.write(path, audio, fs, format='WAV')
        print(f'saved {file_name}.')