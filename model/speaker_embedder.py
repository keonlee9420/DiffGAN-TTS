import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeaker import embedding


class PreDefinedEmbedder(nn.Module):
    """ Speaker Embedder Wrapper """

    def __init__(self, config):
        super(PreDefinedEmbedder, self).__init__()
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]
        self.embedder_type = config["preprocessing"]["speaker_embedder"]
        self.embedder_cuda = config["preprocessing"]["speaker_embedder_cuda"]
        self.embedder = self._get_speaker_embedder()

    def _get_speaker_embedder(self):
        embedder = None
        if self.embedder_type == "DeepSpeaker":
            embedder = embedding.build_model(
                "./deepspeaker/pretrained_models/ResCNN_triplet_training_checkpoint_265.h5"
            )
        else:
            raise NotImplementedError
        return embedder

    def forward(self, audio):
        if self.embedder_type == "DeepSpeaker":
            spker_embed = embedding.predict_embedding(
                self.embedder,
                audio,
                self.sampling_rate,
                self.win_length,
                self.embedder_cuda
            )

        return spker_embed
