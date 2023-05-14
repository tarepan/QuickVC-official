"""Train QuickVC"""

import os
from logging import Logger

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.distributed

from pqmf import PQMF
import commons
import utils
from utils import QuickVCParams
from data_utils_new_new import UnitAudioSpecLoader, TextAudioSpeakerCollate, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss, subband_stft_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch


def run():
    """Training runner"""

    # Backward-compatibility
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '65520'
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

    # Arguments and hyper parameters
    hps = utils.get_hparams()

    # Logger
    logger = utils.get_logger(hps.model_dir)
    writer      = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # Seed
    torch.manual_seed(hps.train.seed)  # pyright: ignore [reportUnknownMemberType]; because of PyTorch

    logger.info(hps)

    # Data
    train_dataset, eval_dataset = UnitAudioSpecLoader("train", hps), UnitAudioSpecLoader("eval", hps)
    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size,
        [32,40,50,60,70,80,90,100,110,120,160,200,230,260,300,350,400,450,500,600,700,800,900,1000], shuffle=True)
    train_collate = TextAudioSpeakerCollate(hps)
    train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False,               pin_memory=True, collate_fn=train_collate, batch_sampler=train_sampler)
    eval_loader  = DataLoader(eval_dataset,  num_workers=2, shuffle=True,  batch_size=1, pin_memory=False, drop_last=False)

    # Model
    ##                          n_freq from n_fft         ,             frame-scale segment size
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
    net_d = MultiPeriodDiscriminator().cuda()
    optim_g = AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)

    # Training state
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str, global_step = 1, 0

    # Sched
    scheduler_g = ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    # AMP
    scaler = GradScaler(enabled=hps.train.fp16_run)

    # Train & Eval
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(global_step, epoch, hps, (net_g, net_d), (optim_g, optim_d), scaler, (train_loader, eval_loader), logger, (writer, writer_eval))
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    global_step: int,
    epoch:       int,
    hps:         QuickVCParams,
    nets:        tuple[SynthesizerTrn, MultiPeriodDiscriminator],
    optims:      tuple[AdamW, AdamW],
    scaler:      GradScaler,
    loaders:     tuple[DataLoader, DataLoader],
    logger:      Logger,
    writers:     tuple[SummaryWriter, SummaryWriter]
):
    """Train and Evaluate QuickVC."""

    #### Epoch #####################################################################################################
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    net_g.train()
    net_d.train()
    for batch_idx, (c, spec, y) in enumerate(train_loader):
        #### Step ################################################################################################
        # Data - Unit series, Linear spectrogram, Waveform
        c, spec, y = c.cuda(non_blocking=True), spec.cuda(non_blocking=True), y.cuda(non_blocking=True)

        mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
        with autocast(enabled=hps.train.fp16_run):
            # G_Forward
            y_hat, y_hat_mb, ids_slice, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(c, spec, mel)
            # G_Loss_1/2
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), 
                hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # D_Forward
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            # D_Loss
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        # D_Backward
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        # D_Optim
        scaler.step(optim_d)


        with autocast(enabled=hps.train.fp16_run):
            # G_Loss_2/2
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                if hps.model.mb_istft_vits == True:
                    pqmf = PQMF()
                    y_mb = pqmf.analysis(y)
                    loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
                else:
                    loss_subband = torch.tensor(0.0)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_subband
        # G_Backward
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        # G_Optim
        scaler.step(optim_g)

        scaler.update()

        # Training logging
        if global_step % hps.train.log_interval == 0:
            # Gradient norm counting
            grad_norm_g = commons.count_grad_norm(net_g.parameters())
            grad_norm_d = commons.count_grad_norm(net_d.parameters())

            lr = optim_g.param_groups[0]['lr']
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, loss_subband]
            logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100. * batch_idx / len(train_loader)))
            logger.info([x.item() for x in losses] + [global_step, lr])
            scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
            scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel,  "loss/g/kl": loss_kl, "loss/g/subband": loss_subband})
            scalar_dict.update({"loss/g/{}".format(i):   v for i, v in enumerate(losses_gen)})
            scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
            image_dict = { 
                "slice/mel_org": utils.plot_spectrogram_to_numpy(    y_mel[0].data.cpu().numpy()),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
                "all/mel":       utils.plot_spectrogram_to_numpy(      mel[0].data.cpu().numpy()),
            }
            utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict)
        # Eval
        if global_step % hps.train.eval_interval == 0:
            evaluate(global_step, hps, net_g, eval_loader, writer_eval)
        # Checkpointing
        if global_step % hps.train.eval_interval == 0:
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))

        global_step += 1
        #### /Step ###############################################################################################

    logger.info(f'====> Epoch: {epoch}')
    #### /Epoch ####################################################################################################


def evaluate(global_step: int, hps: QuickVCParams, net_g: SynthesizerTrn, loader: DataLoader, writer: SummaryWriter):
    """Evaluate reconstruction, then log."""

    net_g.eval()

    # Inference
    with torch.no_grad():
        # Data - only the first sample
        for c, spec, y in loader:
            c, spec, y = c[:1].cuda(), spec[:1].cuda(), y[:1].cuda()
            break
        # Forward
        mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
        y_hat = net_g.infer(c, mel)
        y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1).float(),
          hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)

    # Log
    image_dict = {"gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()), "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
    audio_dict = {"gen/audio": y_hat[0],                                                  "gt/audio": y[0]}
    utils.summarize(writer=writer, global_step=global_step, images=image_dict, audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)

    net_g.train()


if __name__ == "__main__":
    run()
