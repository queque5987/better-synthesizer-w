from pathlib import Path
from models import tacotron
from hparams import hparams
from symbols import symbols
import collections
import os
import torch

class tacotron:
    def __init__(self):
        self._model = tacotron.Tacotron(
            embed_dims=hparams.tts_embed_dims,
            num_chars=len(symbols),
            encoder_dims=hparams.tts_encoder_dims,
            decoder_dims=hparams.tts_decoder_dims,
            n_mels=hparams.num_mels,
            fft_bins=hparams.num_mels,
            postnet_dims=hparams.tts_postnet_dims,
            encoder_K=hparams.tts_encoder_K,
            lstm_dims=hparams.tts_lstm_dims,
            postnet_K=hparams.tts_postnet_K,
            num_highways=hparams.tts_num_highways,
            dropout=hparams.tts_dropout,
            stop_threshold=hparams.tts_stop_threshold,
            speaker_embedding_size=hparams.speaker_embedding_size).to('cpu')

    def load_model(self):
        path = Path("checkpoint/")
        checkpoint = collections.OrderedDict()
        for pt in os.listdir(path):
            tensor = torch.load(os.path.join(path, pt))
            checkpoint[pt[4:-3]] = tensor
        self._model.model_state_load(checkpoint)
        self._model.eval()

    def generate(self, chars, speaker_embeddings):
        return self.generate(chars, speaker_embeddings)

# if __name__ == "__main__":
#     embed = api_test.get_embed()
#     spec = inference(embed, "I love you")
#     print(spec)
    # args = rtvc_args()
    # synthesizer = Synthesizer(args.syn_model_fpath)
    # text = ""
    # embed = []
    # # The synthesizer works in batch, so you need to put your data in a list or numpy array
    # texts = [text]
    # embeds = [embed]
    # # If you know what the attention layer alignments are, you can retrieve them here by
    # # passing return_alignments=True
    # specs = synthesizer.synthesize_spectrograms(texts, embeds)
    # spec = specs[0]
    # print("Created the mel spectrogram")
    # ## Generating the waveform
    # print("Synthesizing the waveform:")