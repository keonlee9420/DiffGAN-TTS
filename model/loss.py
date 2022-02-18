import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from utils.tools import ssim
from utils.pitch_tools import cwt2f0_norm
from text import sil_phonemes_ids


def get_lsgan_losses_fn():

    def jcu_loss_fn(logit_cond, logit_uncond, label_fn, mask=None):
        cond_loss = F.mse_loss(logit_cond, label_fn(logit_cond), reduction="none" if mask is not None else "mean")
        cond_loss = (cond_loss * mask).sum() / mask.sum() if mask is not None else cond_loss
        uncond_loss = F.mse_loss(logit_uncond, label_fn(logit_uncond), reduction="none" if mask is not None else "mean")
        uncond_loss = (uncond_loss * mask).sum() / mask.sum() if mask is not None else uncond_loss
        return 0.5 * (cond_loss + uncond_loss)

    def d_loss_fn(r_logit_cond, r_logit_uncond, f_logit_cond, f_logit_uncond, mask=None):
        r_loss = jcu_loss_fn(r_logit_cond, r_logit_uncond, torch.ones_like, mask)
        f_loss = jcu_loss_fn(f_logit_cond, f_logit_uncond, torch.zeros_like, mask)
        return r_loss, f_loss

    def g_loss_fn(f_logit_cond, f_logit_uncond, mask=None):
        f_loss = jcu_loss_fn(f_logit_cond, f_logit_uncond, torch.ones_like, mask)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'lsgan':
        return get_lsgan_losses_fn()
    else:
        raise NotImplementedError


class DiffGANTTSLoss(nn.Module):
    """ DiffGAN-TTS Loss """

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(DiffGANTTSLoss, self).__init__()
        self.model = args.model
        self.loss_config = train_config["loss"]
        self.pitch_config = preprocess_config["preprocessing"]["pitch"]
        self.pitch_type = self.pitch_config["pitch_type"]
        self.use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
        self.use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.n_layers = model_config["discriminator"]["n_layer"] + \
                        model_config["discriminator"]["n_cond_layer"]
        self.lambda_d = train_config["loss"]["lambda_d"]
        self.lambda_p = train_config["loss"]["lambda_p"]
        self.lambda_e = train_config["loss"]["lambda_e"]
        self.lambda_fm = train_config["loss"]["lambda_fm" if self.model != "shallow" else "lambda_fm_shallow"]
        self.sil_ph_ids = sil_phonemes_ids()
        self.d_loss_fn, self.g_loss_fn = get_adversarial_losses_fn(train_config["loss"]["adv_loss_mode"])

    def forward(self, model, inputs, predictions, coarse_mels=None, Ds=None):
        (
            texts,
            _,
            _,
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            mel2phs,
            _,
        ) = inputs[3:]
        (
            mel_predictions,
            _,
            _,
            _,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        self.src_masks = ~src_masks
        mel_masks = ~mel_masks
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        self.mel_masks = mel_masks[:, :mel_masks.shape[1]]
        self.mel_masks_fill = ~self.mel_masks
        self.mel2phs = mel2phs

        # Acoustic reconstruction loss
        if self.model == "aux":
            mel_loss = torch.zeros(1).to(mel_targets.device)
            for _mel_predictions in mel_predictions:
                _mel_predictions = model.module.diffusion.denorm_spec(_mel_predictions) # since we normalize mel in diffuse_trace
                mel_loss += self.get_mel_loss(_mel_predictions, mel_targets)
        elif self.model == "shallow":
            coarse_mels = coarse_mels[:, : mel_masks.shape[1], :]
            mel_predictions = model.module.diffusion.denorm_spec(mel_predictions) # since we use normalized mel
            mel_loss = self.get_mel_loss(mel_predictions, coarse_mels.detach())
        elif self.model == "naive":
            assert coarse_mels is None
            mel_predictions = model.module.diffusion.denorm_spec(mel_predictions) # since we use normalized mel
            mel_loss = self.get_mel_loss(mel_predictions, mel_targets)

        duration_loss, pitch_loss, energy_loss = self.get_init_losses(mel_targets.device)
        if self.model != "shallow":
            duration_loss = self.get_duration_loss(log_duration_predictions, duration_targets, texts)
            if self.use_pitch_embed:
                pitch_loss = self.get_pitch_loss(pitch_predictions, pitch_targets)
            if self.use_energy_embed:
                energy_loss = self.get_energy_loss(energy_predictions, energy_targets)
        recon_loss = mel_loss + self.lambda_d * sum(duration_loss.values()) + \
                        self.lambda_p * sum(pitch_loss.values()) + self.lambda_e * energy_loss

        # Feature matching loss
        fm_loss = torch.zeros(1).to(mel_targets.device)
        if Ds is not None:
            fm_loss = self.lambda_fm * self.get_fm_loss(*Ds)
            # self.lambda_fm = recon_loss.item() / fm_loss.item() # dynamic scaling following (Yang et al., 2021)

        return (
            fm_loss,
            recon_loss,
            mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )

    def get_init_losses(self, device):
        duration_loss = {
            "pdur": torch.zeros(1).to(device),
            "wdur": torch.zeros(1).to(device),
            "sdur": torch.zeros(1).to(device),
        }
        pitch_loss = {}
        if self.pitch_type == "ph":
            pitch_loss["f0"] = torch.zeros(1).to(device)
        else:
            if self.pitch_type == "cwt":
                pitch_loss["C"] = torch.zeros(1).to(device)
                if self.pitch_config["use_uv"]:
                    pitch_loss["uv"] = torch.zeros(1).to(device)
                pitch_loss["f0_mean"] = torch.zeros(1).to(device)
                pitch_loss["f0_std"] = torch.zeros(1).to(device)
            elif self.pitch_type == "frame":
                if self.pitch_config["use_uv"]:
                    pitch_loss["uv"] = torch.zeros(1).to(device)
                if self.loss_config["pitch_loss"] in ["l1", "l2"]:
                    pitch_loss["f0"] = torch.zeros(1).to(device)
        energy_loss = torch.zeros(1).to(device)
        return duration_loss, pitch_loss, energy_loss

    def get_fm_loss(self, D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond):
        loss_fm = 0
        feat_weights = 4.0 / (self.n_layers + 1)
        for j in range(len(D_fake_cond) - 1):
            loss_fm += feat_weights * \
                0.5 * (F.l1_loss(D_real_cond[j].detach(), D_fake_cond[j]) + F.l1_loss(D_real_uncond[j].detach(), D_fake_uncond[j]))
        return loss_fm

    def get_mel_loss(self, mel_predictions, mel_targets):
        mel_targets.requires_grad = False
        mel_predictions = mel_predictions.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_targets = mel_targets.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_loss = self.l1_loss(mel_predictions, mel_targets)
        return mel_loss

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction="none")
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        dur_gt.requires_grad = False
        losses = {}
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if self.loss_config["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config["dur_loss"] == "mog":
            return NotImplementedError
        elif self.loss_config["dur_loss"] == "crf":
            # losses["pdur"] = -self.model.dur_predictor.crf(
            #     dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction="mean")
            return NotImplementedError
        losses["pdur"] = losses["pdur"] * self.loss_config["lambda_ph_dur"]

        # use linear scale for sent and word duration
        if self.loss_config["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * self.loss_config["lambda_word_dur"]
        if self.loss_config["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean")
            losses["sdur"] = sdur_loss.mean() * self.loss_config["lambda_sent_dur"]
        return losses

    def get_pitch_loss(self, pitch_predictions, pitch_targets):
        losses = {}
        for _, pitch_target in pitch_targets.items():
            if pitch_target is not None:
                pitch_target.requires_grad = False
        if self.pitch_type == "ph":
            nonpadding = self.src_masks.float()
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(pitch_predictions["pitch_pred"][:, :, 0], pitch_targets["f0"],
                                          reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        else:
            mel2ph = self.mel2phs  # [B, T_s]
            f0 = pitch_targets["f0"]
            uv = pitch_targets["uv"]
            nonpadding = self.mel_masks.float()
            if self.pitch_type == "cwt":
                cwt_spec = pitch_targets[f"cwt_spec"]
                f0_mean = pitch_targets["f0_mean"]
                f0_std = pitch_targets["f0_std"]
                cwt_pred = pitch_predictions["cwt"][:, :, :10]
                f0_mean_pred = pitch_predictions["f0_mean"]
                f0_std_pred = pitch_predictions["f0_std"]
                losses["C"] = self.cwt_loss(cwt_pred, cwt_spec) * self.loss_config["lambda_f0"]
                if self.pitch_config["use_uv"]:
                    assert pitch_predictions["cwt"].shape[-1] == 11
                    uv_pred = pitch_predictions["cwt"][:, :, -1]
                    losses["uv"] = (F.binary_cross_entropy_with_logits(uv_pred, uv, reduction="none") * nonpadding) \
                                    .sum() / nonpadding.sum() * self.loss_config["lambda_uv"]
                losses["f0_mean"] = F.l1_loss(f0_mean_pred, f0_mean) * self.loss_config["lambda_f0"]
                losses["f0_std"] = F.l1_loss(f0_std_pred, f0_std) * self.loss_config["lambda_f0"]
                # if self.loss_config["cwt_add_f0_loss"]:
                #     f0_cwt_ = cwt2f0_norm(cwt_pred, f0_mean_pred, f0_std_pred, mel2ph, self.pitch_config)
                #     self.add_f0_loss(f0_cwt_[:, :, None], f0, uv, losses, nonpadding=nonpadding)
            elif self.pitch_type == "frame":
                self.add_f0_loss(pitch_predictions["pitch_pred"], f0, uv, losses, nonpadding=nonpadding)
        return losses

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if self.pitch_config["use_uv"]:
            assert p_pred[..., 1].shape == uv.shape
            losses["uv"] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_uv"]
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if self.loss_config["pitch_loss"] in ["l1", "l2"]:
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(f0_pred, f0, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        elif self.loss_config["pitch_loss"] == "ssim":
            return NotImplementedError

    def cwt_loss(self, cwt_p, cwt_g):
        if self.loss_config["cwt_loss"] == "l1":
            return F.l1_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "l2":
            return F.mse_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "ssim":
            return self.ssim_loss(cwt_p, cwt_g, 20)

    def get_energy_loss(self, energy_predictions, energy_targets):
        energy_targets.requires_grad = False
        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(self.src_masks)
            energy_targets = energy_targets.masked_select(self.src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(self.mel_masks)
            energy_targets = energy_targets.masked_select(self.mel_masks)
        energy_loss = F.l1_loss(energy_predictions, energy_targets)
        return energy_loss
