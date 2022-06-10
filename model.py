from utils import load_F0, load_vocoder, load_starganv2, speakers, preprocess, compute_style
import torch

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Model(metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        Model.starganv2 = load_starganv2()
        Model.F0_model = load_F0()
        Model.vocoder = load_vocoder()
        Model.speakers = speakers


    def infer(self, audio, speaker):
        '''@lw
        :speaker: the speaker name
        '''

        # @lw: set reference, get the speaker index
        speaker_dicts = {speaker: ('', self.speakers[speaker])}

        # @lw: compute reference embeddings
        reference_embeddings = compute_style(speaker_dicts)

        # conversion
        source = preprocess(audio).to('cuda:0')
        converted_audio = None

        for key, (ref, _) in reference_embeddings.items():
            with torch.no_grad():
                f0_feat = self.F0_model.get_feature_GAN(source.unsqueeze(1))
                out = self.starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

                c = out.transpose(-1, -2).squeeze().to('cuda')
                y_out = self.vocoder.inference(c)
                y_out = y_out.view(-1).cpu()

            converted_audio = y_out.numpy()

        return converted_audio