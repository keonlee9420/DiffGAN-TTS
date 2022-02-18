import argparse
import os

import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, get_netG_params, get_netD_params
from utils.tools import get_configs_of, to_device, log, synth_one_sample
from model import DiffGANTTSLoss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", args, preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, discriminator, optG_fs2, optG, optD, sdlG, sdlD, epoch = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    discriminator = nn.DataParallel(discriminator)
    num_params_G = get_param_num(model)
    num_params_D = get_param_num(discriminator)
    Loss = DiffGANTTSLoss(args, preprocess_config, model_config, train_config).to(device)
    print("Number of DiffGAN-TTS Parameters     :", num_params_G)
    print("          JCUDiscriminator Parameters:", num_params_D)
    print("          All Parameters             :", num_params_G + num_params_D)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step_{}".format(args.model)]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    
    def model_update(model, step, loss, optimizer):
        # Backward
        loss = (loss / grad_acc_step).backward()
        if step % grad_acc_step == 0:
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                if args.model == "aux":

                    # Forward
                    output, p_targets, coarse_mels = model(*(batch[2:]))
                    # Update Batch
                    batch[9] = p_targets

                    (
                        fm_loss,
                        recon_loss,
                        mel_loss,
                        pitch_loss,
                        energy_loss,
                        duration_loss,
                    ) = Loss(
                        model,
                        batch,
                        output,
                    )
                    output[0] = output[0][0] # only x_0 is needed after calculating loss

                    G_loss = recon_loss
                    D_loss = fm_loss = adv_loss = torch.zeros(1).to(device)

                    model_update(model, step, G_loss, optG_fs2)

                else: # args.model in ["naive", "shallow"]

                    #######################
                    # Train Discriminator #
                    #######################

                    # Forward
                    output, *_ = model(*(batch[2:]))

                    xs, spk_emb, t, mel_masks = *(output[1:4]), output[9]
                    x_ts, x_t_prevs, x_t_prev_preds, spk_emb, t = \
                        [x.detach() if x is not None else x for x in (list(xs) + [spk_emb, t])]

                    D_fake_cond, D_fake_uncond = discriminator(x_ts, x_t_prev_preds, spk_emb, t)
                    D_real_cond, D_real_uncond = discriminator(x_ts, x_t_prevs, spk_emb, t)

                    D_loss_real, D_loss_fake = Loss.d_loss_fn(D_real_cond[-1], D_real_uncond[-1], D_fake_cond[-1], D_fake_uncond[-1])

                    D_loss = D_loss_real + D_loss_fake

                    model_update(discriminator, step, D_loss, optD)

                    #######################
                    # Train Generator #
                    #######################

                    # Forward
                    output, p_targets, coarse_mels = model(*(batch[2:]))
                    # Update Batch
                    batch[9] = p_targets

                    (x_ts, x_t_prevs, x_t_prev_preds), spk_emb, t, mel_masks = *(output[1:4]), output[9]

                    D_fake_cond, D_fake_uncond = discriminator(x_ts, x_t_prev_preds, spk_emb, t)
                    D_real_cond, D_real_uncond = discriminator(x_ts, x_t_prevs, spk_emb, t)

                    adv_loss = Loss.g_loss_fn(D_fake_cond[-1], D_fake_uncond[-1])

                    (
                        fm_loss,
                        recon_loss,
                        mel_loss,
                        pitch_loss,
                        energy_loss,
                        duration_loss,
                    ) = Loss(
                        model,
                        batch,
                        output,
                        coarse_mels,
                        (D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond),
                    )

                    G_loss = adv_loss + recon_loss + fm_loss

                    model_update(model, step, G_loss, optG)

                losses = [D_loss + G_loss, D_loss, G_loss, recon_loss, fm_loss, adv_loss, mel_loss, pitch_loss, energy_loss, duration_loss]
                losses_msg = [D_loss + G_loss, D_loss, adv_loss, mel_loss, pitch_loss, energy_loss, duration_loss]

                if step % log_step == 0:
                    losses_msg = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses_msg]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, D_loss: {:.4f}, adv_loss: {:.4f}, mel_loss: {:.4f}, pitch_loss: {:.4f}, energy_loss: {:.4f}, duration_loss: {:.4f}".format(
                        *losses_msg
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses, lr=sdlG.get_last_lr()[-1] if args.model != "aux" else optG_fs2.get_last_lr())

                if step % synth_step == 0:
                    figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        args,
                        batch,
                        output,
                        coarse_mels,
                        vocoder,
                        model_config,
                        preprocess_config,
                        model.module.diffusion,
                    )
                    log(
                        train_logger,
                        step,
                        figs=figs,
                        tag="Training",
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        step,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/reconstructed",
                    )
                    log(
                        train_logger,
                        step,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/synthesized",
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(args, model, discriminator, step, configs, val_logger, vocoder, losses)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "G": model.module.state_dict(),
                            "D": discriminator.module.state_dict(),
                            "optG_fs2": optG_fs2._optimizer.state_dict(),
                            "optG": optG.state_dict(),
                            "optD": optD.state_dict(),
                            "sdlG": sdlG.state_dict(),
                            "sdlD": sdlD.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step >= total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1
        if args.model != "aux":
            sdlG.step()
            sdlD.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=True,
        help="training model type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    if args.model == "shallow":
        assert args.restore_step >= train_config["step"]["total_step_aux"]
    if args.model in ["aux", "shallow"]:
        train_tag = "shallow"
    elif args.model == "naive":
        train_tag = "naive"
    else:
        raise NotImplementedError
    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"]+"_{}{}".format(args.model, path_tag)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(" ---> Type of Modeling:", args.model)
    if model_config["multi_speaker"]:
        print(" ---> Type of Speaker Embedder:", preprocess_config["preprocessing"]["speaker_embedder"])
    print(" ---> Total Batch Size:", int(train_config["optimizer"]["batch_size"]))
    print(" ---> Use Pitch Embed:", model_config["variance_embedding"]["use_pitch_embed"])
    print(" ---> Use Energy Embed:", model_config["variance_embedding"]["use_energy_embed"])
    print(" ---> Path of ckpt:", train_config["path"]["ckpt_path"])
    print(" ---> Path of log:", train_config["path"]["log_path"])
    print(" ---> Path of result:", train_config["path"]["result_path"])
    print("================================================================================================")

    main(args, configs)
