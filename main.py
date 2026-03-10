import os
import random
import argparse
import warnings

import numpy as np
import torch
import torch.backends.cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import SoundEventDataset, summarize_dataset_events
from models import create_acf_sed
from trainer import SEDTrainer
import config as cfg

warnings.filterwarnings('ignore')


# ============================================================
# Utilities
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed=42, deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if deterministic and hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ACF-SED: ALE-Confidence Fusion Sound Event Detection"
    )

    # --- General ---
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing subject folders')
    parser.add_argument('--output_dir', type=str, default=cfg.OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_only', type=str, default=None,
                        help='Path to model checkpoint for evaluation-only mode')

    # --- Data ---
    parser.add_argument('--sample_rate', type=int, default=cfg.SAMPLE_RATE)
    parser.add_argument('--window_size', type=float, default=cfg.WINDOW_SIZE)
    parser.add_argument('--window_stride', type=float, default=cfg.WINDOW_STRIDE,
                        help='Stride for sliding window (seconds, sliding mode only)')
    parser.add_argument('--window_mode', type=str, default=cfg.WINDOW_MODE,
                        choices=['fixed', 'sliding'])
    parser.add_argument('--n_mels', type=int, default=cfg.N_MELS)

    # --- STFT ---
    parser.add_argument('--n_fft', type=int, default=cfg.N_FFT)
    parser.add_argument('--hop_length', type=int, default=cfg.HOP_LENGTH)
    parser.add_argument('--win_length', type=int, default=cfg.WIN_LENGTH)

    # --- Model Architecture ---
    parser.add_argument('--d_model', type=int, default=cfg.D_MODEL)
    parser.add_argument('--nhead', type=int, default=cfg.NHEAD)
    parser.add_argument('--num_layers', type=int, default=cfg.NUM_LAYERS)
    parser.add_argument('--kernel_size', type=int, default=cfg.KERNEL_SIZE)
    parser.add_argument('--dropout', type=float, default=cfg.DROPOUT)
    parser.add_argument('--drop_path_rate', type=float, default=cfg.DROP_PATH_RATE)

    # --- ALE / MRAB ---
    parser.add_argument('--mu_mode', type=str, default=cfg.MU_MODE,
                        choices=['fixed', 'scalar', 'per_freq'],
                        help='NLMS step-size mode: fixed / scalar / per_freq')
    parser.add_argument('--mu_init', type=float, default=cfg.MU_INIT,
                        help='Initial NLMS step size (mu_init)')

    # --- Training ---
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=cfg.LR)
    parser.add_argument('--patience', type=int, default=cfg.PATIENCE)
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--deterministic', type=str2bool, nargs='?',
                        const=True, default=True)

    # --- Augmentation ---
    parser.add_argument('--use_augmentation', type=str2bool, nargs='?',
                        const=True, default=cfg.USE_AUGMENTATION)
    parser.add_argument('--mixup_alpha', type=float, default=cfg.MIXUP_ALPHA)
    parser.add_argument('--filteraug_prob', type=float, default=cfg.FILTERAUG_PROB)

    # --- Separate ALE Training ---
    parser.add_argument('--use_separate_ale_training', type=str2bool, nargs='?',
                        const=True, default=cfg.USE_SEPARATE_ALE_TRAINING,
                        help='Train ALE params and backbone with separate optimizers')
    parser.add_argument('--ale_lr', type=float, default=cfg.ALE_LR)
    parser.add_argument('--ale_update_freq', type=int, default=cfg.ALE_UPDATE_FREQ)

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("ACF-SED: ALE-Confidence Fusion Sound Event Detection")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Data:   {args.data_dir}")
    print(f"Output: {args.output_dir}")

    # ---- Load Dataset ----
    print("\nLoading dataset...")
    dataset = SoundEventDataset(
        data_dir=args.data_dir,
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        window_stride=args.window_stride,
        mode=args.window_mode,
    )
    summarize_dataset_events(dataset)

    # ---- Train / Val / Test Split (80/10/10 subject-level) ----
    subjects = list(set(s['subject_id'] for s in dataset.samples))
    train_subjects, temp_subjects = train_test_split(subjects, test_size=0.2, random_state=args.seed)
    val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=args.seed)

    train_indices = [i for i, s in enumerate(dataset.samples) if s['subject_id'] in train_subjects]
    val_indices   = [i for i, s in enumerate(dataset.samples) if s['subject_id'] in val_subjects]
    test_indices  = [i for i, s in enumerate(dataset.samples) if s['subject_id'] in test_subjects]

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    # Wrap to expose class_names and samples attributes
    test_dataset.class_names = dataset.class_names
    test_dataset.samples = [dataset.samples[i] for i in test_indices]

    print(f"\nSplit: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ---- Build Model ----
    print("\nBuilding ACF-SED model...")
    num_classes = len(dataset.class_names)
    model = create_acf_sed(
        num_classes=num_classes,
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        kernel_size=args.kernel_size,
        mu_mode=args.mu_mode,
        mu_init=args.mu_init,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    # ---- Eval-only mode ----
    if args.eval_only is not None:
        print(f"\nEval-only mode: loading {args.eval_only}")
        state = torch.load(args.eval_only, map_location=args.device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        model = model.to(args.device)

        trainer = SEDTrainer(
            model=model,
            device=args.device,
            use_augmentation=False,
        )
        trainer.output_dir = args.output_dir
        trainer._test_with_metrics(test_loader, test_dataset)
        return

    # ---- Train ----
    trainer = SEDTrainer(
        model=model,
        device=args.device,
        use_augmentation=args.use_augmentation,
        mixup_alpha=args.mixup_alpha,
        filteraug_prob=args.filteraug_prob,
        use_separate_ale_training=args.use_separate_ale_training,
        ale_lr=args.ale_lr,
        ale_update_freq=args.ale_update_freq,
    )

    history, test_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )

    print("\nTraining complete.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
