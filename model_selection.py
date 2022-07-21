import os

import soundfile as sf

from model import Model
from utils import load_starganv2

model = Model()
model_lists = [i for i in os.listdir('./Models/') if i.endswith('pth')]
models_path = os.path.join(os.getcwd(), 'Models')

test_path = os.path.join(os.getcwd(), 'Test')
ref_path = os.path.join(test_path, 'Ref')
conv_pth = os.path.join(test_path, 'Converted')

for m in model_lists:
    if m.endswith('pth'):
        model_path = os.path.join(models_path, m)
        model.starganv2 = load_starganv2(model_path)
    else:
        continue

    for ref in os.listdir(ref_path):
        if ref.endswith('wav'):
            audio_path = os.path.join(ref_path, ref)
            audio, sr = sf.read(audio_path, samplerate=24000)
        else:
            continue

        for idx, sp in model.speakers.items():
            converted_audio = model.infer(audio, idx)
            # epoch-speaker-audio-timestamp.wav
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_name = '_'.join([m.split('.')[0], sp, ref.split('_')[0], timestamp]) + '.wav'
            conv_audio_path = os.path.join(conv_pth, file_name)
            sf.write(conv_audio_path, converted_audio, 24000, format='WAV')
            print(f'saved {file_name}.')
    
    print(f'{m} has completed.\n')