# tools/plot_eval.py
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_cmc_curve(cmc_scores, save_path):
    ranks = np.arange(1, len(cmc_scores) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(ranks, cmc_scores, marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('CMC Curve')
    plt.grid(True)
    for r in [1, 5, 10]:
        if r <= len(cmc_scores):
            plt.scatter(r, cmc_scores[r-1], color='red')
            plt.text(r, cmc_scores[r-1], f'Rank-{r}: {cmc_scores[r-1]:.2%}', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Saved CMC curve to", save_path)

def plot_bar(values, labels, ylabel, title, save_path, colors=None):
    plt.figure(figsize=(4+len(values),4))
    colors = colors if colors is not None else ['skyblue']*len(values)
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(values)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Saved bar plot to", save_path)

def plot_training_curves(tb_log_dir, save_dir):
    ea = event_accumulator.EventAccumulator(tb_log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    def plot_tag(tag, ylabel, fname):
        if tag not in tags:
            print(f"Tag {tag} not in TB logs")
            return
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        plt.figure(figsize=(6,4))
        plt.plot(steps, vals, marker='o')
        plt.xlabel('Step/Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} over time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()
        print("Saved training curve", fname)
    plot_tag('loss/loss', 'Loss', 'loss_curve.png')
    plot_tag('val scores/mAP', 'mAP', 'map_curve.png')
    plot_tag('val scores/Rank1', 'Rank-1', 'rank1_curve.png')

def main(exp_dir):
    metrics_dir = os.path.join(exp_dir, 'metrics')
    if not os.path.isdir(metrics_dir):
        print("Metrics dir not found:", metrics_dir); return

    latest = os.path.join(metrics_dir, 'latest_val.npz')
    if not os.path.exists(latest):
        # try to pick the last epoch file
        files = sorted(glob.glob(os.path.join(metrics_dir, 'val_epoch_*.npz')))
        if not files:
            print("No validation npz files found in", metrics_dir); return
        latest = files[-1]

    data = np.load(latest, allow_pickle=True)
    cmc = data['cmc']
    map_v = float(data['mAP'])
    minp = float(data['mINP'])

    out_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    plot_cmc_curve(cmc, os.path.join(out_dir, 'cmc_curve.png'))
    plot_bar([map_v], ['Ours'], 'mAP', 'mAP Comparison', os.path.join(out_dir, 'map_bar.png'))
    plot_bar([minp], ['Ours'], 'mINP', 'mINP Comparison', os.path.join(out_dir, 'minp_bar.png'))

    tb_dir = os.path.join(exp_dir, 'tensorboard')
    if os.path.isdir(tb_dir):
        plot_training_curves(tb_dir, out_dir)
    else:
        print("Tensorboard dir not found, skipping training curves:", tb_dir)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_eval.py <exp_dir>")
        sys.exit(1)
    main(sys.argv[1])
