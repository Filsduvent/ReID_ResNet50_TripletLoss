# train.py
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import json

from tri_loss.dataset import create_dataset
from tri_loss.model.Model import Model
from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.model.loss import global_loss

from tri_loss.utils.utils import time_str
from tri_loss.utils.utils import str2bool
from tri_loss.utils.utils import tight_float_str as tfs
from tri_loss.utils.utils import may_set_mode
from tri_loss.utils.utils import load_state_dict
from tri_loss.utils.utils import load_ckpt
from tri_loss.utils.utils import save_ckpt
from tri_loss.utils.utils import set_devices
from tri_loss.utils.utils import AverageMeter
from tri_loss.utils.utils import to_scalar
from tri_loss.utils.utils import ReDirectSTD
from tri_loss.utils.utils import set_seed
from tri_loss.utils.utils import adjust_lr_exp
from tri_loss.utils.utils import adjust_lr_staircase
from tri_loss.utils.utils import may_make_dir  # <<< used for ensuring dirs

# <<< ADDED imports
import shutil

class Config(object):
  def __init__(self):

    #1.Parsing Command Line Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Image preprocessing Parameters
    parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--mirror', type=str2bool, default=True)
    parser.add_argument('--ids_per_batch', type=int, default=32)
    parser.add_argument('--ims_per_id', type=int, default=4)

    # Logging and evaluation configs
    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=20)
    parser.add_argument('--epochs_per_val', type=int, default=1)  # run val each epoch by default

    # Model Architecture + Loss parameters
    parser.add_argument('--last_conv_stride', type=int, default=1,
                        choices=[1, 2])
    parser.add_argument('--normalize_feature', type=str2bool, default=False)
    parser.add_argument('--margin', type=float, default=0.3)

    # Execution Flow Controls
    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    # Learning Rate and Training Hyperparams
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp',
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=151)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(101, 201,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=300)

    # <<< New args for checkpointing and early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=30,
                        help='Stop training if no improvement on validation mAP for this many validation checks (epochs). Set 0 to disable.')
    parser.add_argument('--save_best_only', type=str2bool, default=True,
                        help='Only keep last + best checkpoints when True.')
    parser.add_argument('--save_every_n_epochs', type=int, default=1,
                        help='How often (in epochs) to save checkpoint. 1 = every epoch.')

    args = parser.parse_args()

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    # If you want to make your results exactly reproducible, you have
    # to fix a random seed.
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    # Image Processing
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.train_mirror_type = 'random' if args.mirror else None

    self.ids_per_batch = args.ids_per_batch
    self.ims_per_id = args.ims_per_id
    self.train_final_batch = False
    self.train_shuffle = True

    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = None
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      ids_per_batch=self.ids_per_batch,
      ims_per_id=self.ims_per_id,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.val_set_kwargs = dict(
      part='val',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.val_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    self.last_conv_stride = args.last_conv_stride
    self.normalize_feature = args.normalize_feature
    self.margin = args.margin

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    self.total_epochs = args.total_epochs
    self.epochs_per_val = args.epochs_per_val
    self.steps_per_log = args.steps_per_log
    self.only_test = args.only_test
    self.resume = args.resume

    #######
    # Log #
    #######

    self.log_to_file = args.log_to_file

    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        #
        'lcs_{}_'.format(self.last_conv_stride) +
        ('nf_' if self.normalize_feature else 'not_nf_') +
        'margin_{}_'.format(tfs(self.margin)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file

    # <<< New checkpointing / early stopping params
    self.early_stopping_patience = args.early_stopping_patience
    self.save_best_only = args.save_best_only
    self.save_every_n_epochs = args.save_every_n_epochs


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    feat = self.model(ims)
    feat = feat.data.cpu().numpy()
    self.model.train(old_train_eval_model)
    return feat


def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    may_make_dir(cfg.exp_dir)  # <<< ensure exp dir exists before redirecting
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Save config to json for reproducibility
  try:
    cfg_json_path = osp.join(cfg.exp_dir, 'config.json')
    may_make_dir(osp.dirname(cfg_json_path))
    with open(cfg_json_path, 'w') as f:
      json.dump(cfg.__dict__, f, indent=2, sort_keys=True)
    print(f"Saved config to {cfg_json_path}")
  except Exception as e:
    print("Warning: could not save config.json:", e)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########

  if not cfg.only_test:
    train_set = create_dataset(**cfg.train_set_kwargs)
    # The combined dataset does not provide val set currently.
    val_set = None if cfg.dataset == 'combined' else create_dataset(**cfg.val_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  model = Model(last_conv_stride=cfg.last_conv_stride)
  # Model wrapper
  model_w = DataParallel(model)

  #############################
  # Criteria and Optimizers   #
  #############################

  tri_loss = TripletLoss(margin=cfg.margin)

  optimizer = optim.Adam(model.parameters(),
                         lr=cfg.base_lr,
                         weight_decay=cfg.weight_decay)

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  resume_ep = 0
  best_val_map = -1.0  # <<< track best mAP
  best_ckpt_file = osp.join(cfg.exp_dir, 'ckpt_best.pth')  # <<< best ckpt path
  epochs_no_improve = 0  # <<< early stopping counter

  if cfg.resume:
    try:
      resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)
      resume_ep = int(resume_ep)
      print(f"Resuming from epoch {resume_ep}")
    except Exception as e:
      print("Warning: resume requested but failed to load ckpt:", e)
      resume_ep = 0

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

  # make sure metrics dir exists
  metrics_dir = osp.join(cfg.exp_dir, 'metrics')
  may_make_dir(metrics_dir)

  ########
  # Test #
  ########

  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        load_ckpt(modules_optims, cfg.ckpt_file)

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      # test_set.eval prints results
      mAP, cmc_scores, mINP, mq_mAP, mq_cmc, mq_mINP = test_set.eval(
        normalize_feat=cfg.normalize_feature,
        verbose=True)
      # Save the test metrics to file
      out_file = osp.join(cfg.exp_dir, 'metrics', f'test_{name}.npz')
      np.savez(out_file, cmc=cmc_scores, mAP=mAP, mINP=mINP)
      print(f"Saved test metrics to {out_file}")

  def validate(ep):
    """Run validation and save per-epoch metrics; returns key metrics."""
    if val_set.extract_feat_func is None:
      val_set.set_feat_func(ExtractFeature(model_w, TVT))
    print('\n=========> Test on validation set <=========\n')
    # Unpack mAP, cmc_scores, mINP (ignore multi-query for val)
    mAP, cmc_scores, mINP, *_ = val_set.eval(
        normalize_feat=cfg.normalize_feature,
        to_re_rank=False,
        verbose=False)
    # === Save results to .npz for later plotting ===
    metrics_dir_local = osp.join(cfg.exp_dir, 'metrics')
    may_make_dir(metrics_dir_local)
    eval_file = osp.join(metrics_dir_local, f'val_epoch_{ep+1:03d}.npz')
    np.savez(eval_file, cmc=cmc_scores, mAP=mAP, mINP=mINP)
    # also save latest snapshot of metrics
    latest_file = osp.join(metrics_dir_local, 'latest_val.npz')
    np.savez(latest_file, cmc=cmc_scores, mAP=mAP, mINP=mINP)
    print(f"Saved validation metrics to {eval_file} and latest_val.npz")
    print()
    # Return mAP, Rank-1, Rank-5, Rank-10, mINP
    return mAP, cmc_scores[0], cmc_scores[4], cmc_scores[9], mINP

  if cfg.only_test:
    test(load_model_weight=True)
    return

  ############
  # Training #
  ############

  start_ep = resume_ep if cfg.resume else 0
  # If resuming, we might want to set best_val_map if ckpt has scores info -- try to load it
  # NOTE: load_ckpt returns (ep, scores); original code didn't use scores. Here we keep simple.

  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    if cfg.lr_decay_type == 'exp':
      adjust_lr_exp(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.total_epochs,
        cfg.exp_decay_at_epoch)
    else:
      adjust_lr_staircase(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiply_factor)

    may_set_mode(modules_optims, 'train')

    # For recording precision, satisfying margin, etc
    prec_meter = AverageMeter()
    sm_meter = AverageMeter()
    dist_ap_meter = AverageMeter()
    dist_an_meter = AverageMeter()
    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_t = TVT(torch.from_numpy(labels).long())

      feat = model_w(ims_var)

      loss, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat, labels_t,
        normalize_feature=cfg.normalize_feature)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ############
      # Step Log #
      ############

      # precision
      prec = (dist_an > dist_ap).data.float().mean()
      # the proportion of triplets that satisfy margin
      sm = (dist_an > dist_ap + cfg.margin).data.float().mean()
      # average (anchor, positive) distance
      d_ap = dist_ap.data.mean()
      # average (anchor, negative) distance
      d_an = dist_an.data.mean()

      prec_meter.update(prec)
      sm_meter.update(sm)
      dist_ap_meter.update(d_ap)
      dist_an_meter.update(d_an)
      loss_meter.update(to_scalar(loss))

      if step % cfg.steps_per_log == 0:
        time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
          step, ep + 1, time.time() - step_st, )

        tri_log = (', prec {:.2%}, sm {:.2%}, '
                   'd_ap {:.4f}, d_an {:.4f}, '
                   'loss {:.4f}'.format(
          prec_meter.val, sm_meter.val,
          dist_ap_meter.val, dist_an_meter.val,
          loss_meter.val, ))

        log = time_log + tri_log
        print(log)

    #############
    # Epoch Log #
    #############

    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st)

    tri_log = (', prec {:.2%}, sm {:.2%}, '
               'd_ap {:.4f}, d_an {:.4f}, '
               'loss {:.4f}'.format(
      prec_meter.avg, sm_meter.avg,
      dist_ap_meter.avg, dist_an_meter.avg,
      loss_meter.avg, ))

    log = time_log + tri_log
    print(log)

    ##########################
    # Test on Validation Set #
    ##########################

    mAP, Rank1, Rank5, Rank10, mINP = 0, 0, 0, 0, 0
    if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
      mAP, Rank1, Rank5, Rank10, mINP = validate(ep)
      print(f"Validation: mAP={mAP:.4%}, Rank-1={Rank1:.4%}, Rank-5={Rank5:.4%}, Rank-10={Rank10:.4%}, mINP={mINP:.4%}")

      # Check for new best
      is_new_best = False
      if mAP > best_val_map:
        is_new_best = True
        best_val_map = mAP
        epochs_no_improve = 0
      else:
        epochs_no_improve += 1

      # Save best checkpoint if improved
      if cfg.log_to_file and is_new_best:
        print(f"New best mAP: {best_val_map:.6f} -> saving best checkpoint to {best_ckpt_file}")
        save_ckpt(modules_optims, ep + 1, best_val_map, best_ckpt_file)

      # Early stopping check
      if cfg.early_stopping_patience > 0 and epochs_no_improve >= cfg.early_stopping_patience:
        print(f"Early stopping triggered. No improvement for {epochs_no_improve} validation checks (patience={cfg.early_stopping_patience}).")
        # Save one last checkpoint (last)
        if cfg.log_to_file:
          save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)
        break

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(mAP=mAP,
             Rank1=Rank1,
             Rank5=Rank5,
             Rank10=Rank10,
             mINP=mINP),
        ep)
      writer.add_scalars(
        'loss',
        dict(loss=loss_meter.avg, ),
        ep)
      writer.add_scalars(
        'precision',
        dict(precision=prec_meter.avg, ),
        ep)
      writer.add_scalars(
        'satisfy_margin',
        dict(satisfy_margin=sm_meter.avg, ),
        ep)
      writer.add_scalars(
        'average_distance',
        dict(dist_ap=dist_ap_meter.avg,
             dist_an=dist_an_meter.avg, ),
        ep)

    # save ckpt (last)
    if cfg.log_to_file and ((ep + 1) % cfg.save_every_n_epochs == 0):
      # Save last ckpt
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)
      # If not saving history, keep only last + best (best saved separately)
      if cfg.save_best_only:
        # nothing extra to remove because we overwrite ckpt_file each time
        pass

  ########
  # Test #
  ########

  # If we exited loop via early stopping, still run final test with best weights if exists
  if osp.exists(best_ckpt_file):
    print("Testing using best checkpoint:", best_ckpt_file)
    # load best
    map_location = (lambda storage, loc: storage)
    try:
      loaded = torch.load(best_ckpt_file, map_location=map_location)
      # load_state_dict expects the state_dict or dict of state_dicts; save_ckpt saved similar structure
      # load_ckpt logic handles modules_optims; reuse it
      load_ckpt(modules_optims, best_ckpt_file)
    except Exception as e:
      print("Warning: failed to load best ckpt for test:", e)

  test(load_model_weight=False)


if __name__ == '__main__':
  main()
