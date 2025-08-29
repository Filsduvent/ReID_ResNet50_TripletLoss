import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# For reading TensorBoard logs
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    raise ImportError("Please install tensorboard: pip install tensorboard")

def plot_cmc_curve(cmc_scores, save_path='cmc_curve.png'):
    ranks = np.arange(1, len(cmc_scores) + 1)
    plt.figure()
    plt.plot(ranks, cmc_scores, marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('CMC Curve')
    plt.grid(True)
    # mark Rank-1/5/10 if available
    for r in [1, 5, 10]:
        if r <= len(cmc_scores):
            plt.scatter(r, cmc_scores[r-1], color='red')
            plt.text(r, cmc_scores[r-1], f'Rank-{r}: {cmc_scores[r-1]:.2%}', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved CMC curve to {save_path}")

def plot_bar(values, labels, ylabel, title, save_path, color='skyblue'):
    plt.figure()
    plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, min(0.98, v + 0.01), f'{v:.2%}', ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved bar plot to {save_path}")

def plot_training_curves(tb_log_dir, save_dir):
    if not os.path.exists(tb_log_dir):
        print(f"TensorBoard log dir not found: {tb_log_dir}")
        return
    ea = event_accumulator.EventAccumulator(tb_log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])

    def plot_scalar(tag, ylabel, save_name):
        if tag not in tags:
            print(f"Tag {tag} not found in TensorBoard logs.")
            return
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        plt.figure()
        plt.plot(steps, vals, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} over Epochs')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.close()
        print(f"Saved training curve {save_name}")

    # These tags match what your train.py writes
    plot_scalar('loss/loss', 'Loss', 'loss_curve.png')
    plot_scalar('val scores/mAP', 'mAP', 'map_curve.png')
    plot_scalar('val scores/Rank1', 'Rank-1', 'rank1_curve.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the experiment directory (the one that contains metrics/ and tensorboard/)')
    args = parser.parse_args()

    # === 1. Define paths ===
    exp_dir = args.exp_dir
    metrics_dir = os.path.join(exp_dir, 'metrics')
    tb_log_dir = os.path.join(exp_dir, 'tensorboard')
    save_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    # === 2. Find latest validation metrics ===
    if not os.path.isdir(metrics_dir):
        print(f"No metrics dir found at {metrics_dir}")
        return

    metric_files = sorted([f for f in os.listdir(metrics_dir) if f.endswith('.npz')])
    if not metric_files:
        print(f"No metrics found in {metrics_dir}")
        return

    latest_metric_file = os.path.join(metrics_dir, metric_files[-1])
    print(f"Using metrics file: {latest_metric_file}")

    results = np.load(latest_metric_file, allow_pickle=True)
    cmc_scores = results['cmc']  # shape: [num_ranks]

    # mAP / mINP may be numpy scalars or plain floats
    def to_float(x):
        try:
            return float(x.item())
        except Exception:
            return float(x)

    mAP = to_float(results['mAP'])
    mINP = to_float(results['mINP'])

    # === 3. Plot CMC Curve ===
    plot_cmc_curve(cmc_scores, save_path=os.path.join(save_dir, 'cmc_curve.png'))

    # === 4. Plot mAP and mINP bar plots ===
    plot_bar([mAP], ['Latest'], 'mAP', 'mAP (Latest Validation)', os.path.join(save_dir, 'map_barplot.png'), color='skyblue')
    plot_bar([mINP], ['Latest'], 'mINP', 'mINP (Latest Validation)', os.path.join(save_dir, 'minp_barplot.png'), color='orange')

    # === 5. Plot training curves from TensorBoard ===
    plot_training_curves(tb_log_dir, save_dir)

if __name__ == '__main__':
    main()











"""import os
import numpy as np
import matplotlib.pyplot as plt

# For reading TensorBoard logs
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    raise ImportError("Please install tensorboard: pip install tensorboard")

def plot_cmc_curve(cmc_scores, save_path='cmc_curve.png'):
    ranks = np.arange(1, len(cmc_scores) + 1)
    plt.figure()
    plt.plot(ranks, cmc_scores, marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('CMC Curve')
    plt.grid(True)
    for r in [1, 5, 10]:
        plt.scatter(r, cmc_scores[r-1], color='red')
        plt.text(r, cmc_scores[r-1], f'Rank-{r}: {cmc_scores[r-1]:.2%}', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved CMC curve to {save_path}")

def plot_bar(values, labels, ylabel, title, save_path, color='skyblue'):
    plt.figure()
    plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved bar plot to {save_path}")

def plot_training_curves(tb_log_dir, save_dir):
    ea = event_accumulator.EventAccumulator(tb_log_dir)
    ea.Reload()
    tags = ea.Tags()['scalars']

    def plot_scalar(tag, ylabel, save_name):
        if tag not in tags:
            print(f"Tag {tag} not found in TensorBoard logs.")
            return
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        plt.figure()
        plt.plot(steps, vals, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} over Epochs')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.close()
        print(f"Saved training curve {save_name}")

    plot_scalar('loss/loss', 'Loss', 'loss_curve.png')
    plot_scalar('val scores/mAP', 'mAP', 'map_curve.png')
    plot_scalar('val scores/Rank1', 'Rank-1', 'rank1_curve.png')

def main():
    # === 1. Define paths ===
    exp_dir = 'exp/smole_test'  # CHANGE to your experiment folder
    metrics_dir = os.path.join(exp_dir, 'metrics')
    tb_log_dir = os.path.join(exp_dir, 'tensorboard')
    save_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    # === 2. Find latest validation metrics ===
    metric_files = sorted([f for f in os.listdir(metrics_dir) if f.endswith('.npz')])
    if not metric_files:
        print(f"No metrics found in {metrics_dir}")
        return

    latest_metric_file = os.path.join(metrics_dir, metric_files[-1])
    print(f"Using metrics file: {latest_metric_file}")

    results = np.load(latest_metric_file)
    cmc_scores = results['cmc']
    mAP = float(results['mAP'].item()) if hasattr(results['mAP'], 'item') else float(results['mAP'])
    mINP = float(results['mINP'].item()) if hasattr(results['mINP'], 'item') else float(results['mINP'])

    # === 3. Plot CMC Curve ===
    plot_cmc_curve(cmc_scores, save_path=os.path.join(save_dir, 'cmc_curve.png'))

    # === 4. Plot mAP and mINP bar plots ===
    plot_bar([mAP], ['Latest'], 'mAP', 'mAP Comparison', os.path.join(save_dir, 'map_barplot.png'), color='skyblue')
    plot_bar([mINP], ['Latest'], 'mINP', 'mINP Comparison', os.path.join(save_dir, 'minp_barplot.png'), color='orange')

    # === 5. Plot training curves from TensorBoard ===
    plot_training_curves(tb_log_dir, save_dir)

if __name__ == '__main__':
    main()"""