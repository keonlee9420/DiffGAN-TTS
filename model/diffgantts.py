import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LinearNorm, ConvNorm, DiffusionEmbedding, Mish
from .modules import FastspeechEncoder, FastspeechDecoder, VarianceAdaptor
from .diffusion import GaussianDiffusion
from utils.tools import get_mask_from_lengths
from .loss import get_adversarial_losses_fn


class DiffGANTTS(nn.Module):
    """ DiffGAN-TTS """

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(DiffGANTTS, self).__init__()
        self.model = args.model
        self.model_config = model_config

        self.text_encoder = FastspeechEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        if self.model in ["aux", "shallow"]:
            self.decoder = FastspeechDecoder(model_config)
            self.mel_linear = nn.Linear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )
        self.diffusion = GaussianDiffusion(args, preprocess_config, model_config, train_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        mel2phs=None,
        spker_embeds=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.text_encoder(texts, src_masks)

        speaker_emb = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_emb = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_emb = self.speaker_emb(spker_embeds) # [B, H]

        (
            output,
            p_targets,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            max_src_len,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            mel2phs,
            p_control,
            e_control,
            d_control,
            speaker_emb,
        )

        if self.model == "naive":
            coarse_mels = None
            (
                output, # x_0_pred
                x_ts,
                x_t_prevs,
                x_t_prev_preds,
                diffusion_step,
            ) = self.diffusion(
                mels,
                output,
                speaker_emb,
                mel_masks,
            )
        elif self.model in ["aux", "shallow"]:
            x_ts = x_t_prevs = x_t_prev_preds = diffusion_step = None
            cond = output.clone()
            coarse_mels = self.decoder(output, mel_masks)
            coarse_mels = self.mel_linear(coarse_mels)
            if self.model == "aux":
                output = self.diffusion.diffuse_trace(coarse_mels, mel_masks)
            elif self.model == "shallow":
                (
                    output, # x_0_pred
                    x_ts,
                    x_t_prevs,
                    x_t_prev_preds,
                    diffusion_step,
                ) = self.diffusion(
                    mels,
                    self._detach(cond),
                    self._detach(speaker_emb),
                    self._detach(mel_masks),
                    self._detach(coarse_mels),
                )
        else:
            raise NotImplementedError

        return [
            output,
            (x_ts, x_t_prevs, x_t_prev_preds),
            self._detach(speaker_emb),
            diffusion_step,
            p_predictions, # cannot detach each value in dict but no problem since loss will not use it
            self._detach(e_predictions),
            log_d_predictions, # cannot detach each value in dict but no problem since loss will not use it
            self._detach(d_rounded),
            self._detach(src_masks),
            self._detach(mel_masks),
            self._detach(src_lens),
            self._detach(mel_lens),
        ], p_targets, self._detach(coarse_mels)

    def _detach(self, p):
        return p.detach() if p is not None and self.model == "shallow" else p


class JCUDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, preprocess_config, model_config, train_config):
        super(JCUDiscriminator, self).__init__()

        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        n_layer = model_config["discriminator"]["n_layer"]
        n_uncond_layer = model_config["discriminator"]["n_uncond_layer"]
        n_cond_layer = model_config["discriminator"]["n_cond_layer"]
        n_channels = model_config["discriminator"]["n_channels"]
        kernel_sizes = model_config["discriminator"]["kernel_sizes"]
        strides = model_config["discriminator"]["strides"]
        self.multi_speaker = model_config["multi_speaker"]

        self.input_projection = LinearNorm(2 * n_mel_channels, 2 * n_mel_channels)
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, n_channels[n_layer-1]),
        )
        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(
                LinearNorm(residual_channels, n_channels[n_layer-1]),
            )
        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer)
            ]
        )
        self.uncond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_uncond_layer)
            ]
        )
        self.cond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_cond_layer)
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts, x_t_prevs, s, t):
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        t -- [B]
        """
        x = self.input_projection(
            torch.cat([x_t_prevs, x_ts], dim=-1)
        ).transpose(1, 2)
        diffusion_step = self.mlp(self.diffusion_embedding(t)).unsqueeze(-1)
        if self.multi_speaker:
            speaker_emb = self.spk_mlp(s).unsqueeze(-1)

        cond_feats = []
        uncond_feats = []
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)

        x_cond = (x + diffusion_step + speaker_emb) \
            if self.multi_speaker else (x + diffusion_step)
        x_uncond = x

        for layer in self.cond_conv_block:
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
        return cond_feats, uncond_feats
