import sounddevice as sd

# set input and output devices
sd.default.device = 0, 1
fs = 24000
sd.default.samplerate = fs  # set sample rate
sd.default.channels = 1, 2  # one input channel, two output channel

duration = 10  # seconds

print('begin recording')
# record voice
audio = sd.rec(int(duration * fs), dtype='float32')
sd.wait()  # wait to recording
print('begin playing')
sd.play(audio, fs)
sd.wait()
sd.default.samplerate = fs  # specify sample rate
sd.default.dtype = 'float32', 'float32'  # specify data type
sd.default.channels = 1, 2  # specify the input/output channels


def callback(in_data, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    q.put(in_data.copy())


while True:
    audio = np.zeros(shape=(24000, 1), dtype='float32')
    q = queue.Queue()
    # set input and output device
    print(sd.query_devices())
    input_device = int(input('plz type the input device:\t'))
    output_device = int(input('plz type the output device:\t'))
    sd.default.device = input_device, output_device

    try:
        with sd.InputStream(samplerate=24000,
                            dtype='float32',
                            channels=1,
                            callback=callback):
            start_time = time.time()

            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                audio = np.append(audio, q.get(), axis=0)
    except KeyboardInterrupt:
        end_time = time.time()
        print('\nRecording finished: costs {} {}'.format(
            end_time - start_time, np.shape(audio)))
        start_time = time.time()
        sd.play(audio)
        sd.wait()
        end_time = time.time()
        print('\nPlaying finished: costs {}'.format(end_time - start_time))
