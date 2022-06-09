import sounddevice as sd

# set input and output devices
sd.default.device = 3, 5
fs = 24000
sd.default.samplerate = fs # set sample rate
sd.default.channels = 2, 2 # one input channel, two output channel

duration = 10 # seconds

print('begin recording')
# record voice
audio = sd.rec(int(duration * fs), dtype = 'float32')
sd.wait() # wait to recording
print('begin playing')
sd.play(audio, fs)
sd.wait()