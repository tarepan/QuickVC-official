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
from data_utils_new_new import UnitAudioSpecLoader, UnitSpecWaveCollate, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss, subband_stft_loss
from mel_processing import wave_to_mel, spec_to_mel


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
    train_collate = UnitSpecWaveCollate(hps)
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
        global_step = train_and_evaluate(global_step, epoch, hps, (net_g, net_d), (optim_g, optim_d), scaler, (train_loader, eval_loader), logger, (writer, writer_eval))
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
) -> int:
    """Train and Evaluate QuickVC."""

    #### Epoch #####################################################################################################
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    if hps.model.mb_istft_vits:
        pqmf = PQMF()

    net_g.train()
    net_d.train()

    # !pip install torch_tb_profiler
    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
    #         schedule=torch.profiler.schedule(wait=20, warmup=15, active=3, repeat=5),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('/content/gdrive/MyDrive/ML_results/quickvc_official/jvs/amp_b64_05'),
    #         profile_memory=True,
    # ) as prof:

    for batch_idx, (c, spec, y) in enumerate(train_loader):
        #### Step ################################################################################################
        # Data - Unit series :: (B, Feat, Frame), Linear spectrogram :: (B, Freq, Frame), Waveform :: (B, 1, T)
        c, spec, y = c.cuda(non_blocking=True), spec.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Common_Forward
        with autocast(enabled=hps.train.fp16_run):
            # TODO: preprocessing. Loader becomes heavy, but in both case, same size of mel is loaded on GPU memory.)
            mel = spec_to_mel(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_hat, y_hat_mb, ids_slice, (_, z_p, m_p, logs_p, _, logs_q) = net_g(c, spec, mel)
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
        # D_Forward/Loss
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
        # D_Backward
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        # D_Optim
        scaler.step(optim_d)

        # G_Loss
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            y_hat_mel = wave_to_mel(y_hat.squeeze(1), 
                hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            loss_mel = hps.train.c_mel * F.l1_loss(y_mel, y_hat_mel)
            loss_kl  = hps.train.c_kl  * kl_loss(z_p, logs_q, m_p, logs_p)
            loss_fm  =                   feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen =       generator_loss(y_d_hat_g)
            if hps.model.mb_istft_vits:
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
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, loss_subband]
            logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100. * batch_idx / len(train_loader)))
            logger.info([x.item() for x in losses] + [global_step])
            scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all,}
            scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel,  "loss/g/kl": loss_kl, "loss/g/subband": loss_subband})
            scalar_dict.update({f"loss/g/{i}":   v for i, v in enumerate(losses_gen)})
            scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})
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

    return global_step

def evaluate(global_step: int, hps: QuickVCParams, net_g: SynthesizerTrn, loader: DataLoader, writer: SummaryWriter):
    """Evaluate reconstruction, then log."""

    net_g.eval()

    # Inference
    with torch.no_grad():
        # Data - only the first sample
        #   Unit series :: (B=1, Feat, Frame), Linear spectrogram :: (B=1, Freq, Frame), Waveform :: (B=1, 1, T)
        for i, (c, spec, y) in enumerate(loader):
            c, spec, y = c.cuda(), spec.cuda(), y.cuda()
            # Forward
            mel = spec_to_mel(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_hat = net_g.infer(c, mel)
            y_hat_mel = wave_to_mel(
            y_hat.squeeze(1).float(),
            hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)

            # Log
            image_dict = {f"gen/mel_{i}": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()), f"gt/mel_{i}": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
            audio_dict = {f"gen/audio_{i}": y_hat[0],                                                  f"gt/audio_{i}": y[0]}
            utils.summarize(writer=writer, global_step=global_step, images=image_dict, audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)

            if i > 5:
                break

    net_g.train()


if __name__ == "__main__":
    run()
