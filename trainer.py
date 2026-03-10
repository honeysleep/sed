import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils.metrics import SEDMetricsCalculator
from utils.augmentation import apply_filter_aug, apply_balanced_mixup


class SEDTrainer:
    def __init__(
        self,
        model,
        device: str = "cuda",
        use_augmentation: bool = True,
        mixup_alpha: float = 0.4,
        filteraug_prob: float = 0.5,
        use_separate_ale_training: bool = False,
        ale_lr: float = 1e-4,
        ale_update_freq: int = 1,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_augmentation = use_augmentation
        self.mixup_alpha = mixup_alpha
        self.filteraug_prob = filteraug_prob

        self.use_separate_ale_training = use_separate_ale_training
        self.ale_lr = ale_lr
        self.ale_update_freq = ale_update_freq
        self.ale_optimizer = None

        self.criterion = nn.BCELoss()
        self.output_dir = "."

    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        test_dataset,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 20,
        output_dir: str = ".",
        resume_from: str = None,
    ):
        print("\n" + "=" * 80)
        if self.use_augmentation:
            print("Training with Data Augmentation (Mixup & FilterAug)")
            print(f"  Mixup Alpha: {self.mixup_alpha}")
            print(f"  FilterAug Prob: {self.filteraug_prob}")
        else:
            print("Training WITHOUT Data Augmentation")
        if self.use_separate_ale_training:
            print(f"  Separate ALE Training: ENABLED (ale_lr={self.ale_lr})")
        print("=" * 80)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # ---- Setup Optimizers ----
        if self.use_separate_ale_training:
            ale_params = self.model.get_trainable_ale_params()
            backbone_params = self.model.get_non_ale_params()

            if len(ale_params) > 0:
                print(f"  ALE parameters:      {sum(p.numel() for p in ale_params):,}")
                print(f"  Backbone parameters: {sum(p.numel() for p in backbone_params):,}")

                self.ale_optimizer = torch.optim.AdamW(
                    ale_params, lr=self.ale_lr, weight_decay=1e-5, betas=(0.9, 0.999),
                )
                optimizer = torch.optim.AdamW(
                    backbone_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.999),
                )
            else:
                print("  Warning: No ALE parameters found; using standard training")
                self.use_separate_ale_training = False
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999),
                )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999),
            )

        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=0.3, div_factor=25, final_div_factor=1e4
        )

        best_val_f1 = 0.0
        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 0

        history = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "mu": [],
        }

        model_save_path = os.path.join(output_dir, "best_model_acf_sed.pth")
        checkpoint_path = os.path.join(output_dir, "checkpoint_last.pth")

        # ---- Resume from checkpoint ----
        if resume_from is not None:
            print(f"\nResuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)

            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"\nWarning: Checkpoint incompatible: {e}")
                print("   Starting training from scratch.")
                resume_from = None

            if resume_from is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_f1 = checkpoint.get('best_val_f1', 0)
                best_val_loss = checkpoint.get('best_val_loss', float("inf"))
                patience_counter = checkpoint.get('patience_counter', 0)
                history = checkpoint.get('history', history)
                print(f"   Resumed from epoch {start_epoch}, best val F1: {best_val_f1:.4f}")

                if self.use_separate_ale_training and self.ale_optimizer is not None:
                    if 'ale_optimizer_state_dict' in checkpoint:
                        self.ale_optimizer.load_state_dict(checkpoint['ale_optimizer_state_dict'])

        # ---- Training Loop ----
        for epoch in range(start_epoch, epochs):
            train_loss, train_f1 = self._train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_f1 = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)

            if hasattr(self.model, 'get_ale_status'):
                mu_status = self.model.get_ale_status()
                history["mu"].append(mu_status)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch + 1}/{epochs} | LR: {current_lr:.2e}")
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

            # Print ALE mu (trainable modes only)
            if history["mu"] and hasattr(self.model, 'mrab'):
                mu_mode = self.model.mrab.filters[0].mu_mode
                if mu_mode != 'fixed':
                    mu_str = " | ".join(
                        f"{k}={v:.4f}" for k, v in history["mu"][-1].items()
                    )
                    print(f"ALE mu: {mu_str}")

            # Save checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'history': history,
            }
            if self.use_separate_ale_training and self.ale_optimizer is not None:
                checkpoint_state['ale_optimizer_state_dict'] = self.ale_optimizer.state_dict()

            torch.save(checkpoint_state, checkpoint_path)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Best model saved: {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Checkpoint saved: {checkpoint_path}")

        # ---- Final Evaluation ----
        print(f"\nLoading best model from: {model_save_path}")
        self.model.load_state_dict(
            torch.load(model_save_path, map_location=self.device, weights_only=False)
        )
        test_results = self._test_with_metrics(test_loader, test_dataset)

        return history, test_results

    def _train_epoch(self, loader, optimizer, scheduler):
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        pbar = tqdm(loader, desc="Training", leave=False)
        batch_idx = 0

        for batch in pbar:
            audio_orig = batch["audio"].to(self.device)
            labels_orig = batch["labels"].to(self.device)

            if self.use_augmentation:
                audio_aug = apply_filter_aug(audio_orig.clone(), prob=self.filteraug_prob)
                audio_aug, labels_aug = apply_balanced_mixup(audio_aug, labels_orig.clone(), alpha=self.mixup_alpha)
                audio_train = torch.cat([audio_orig, audio_aug], dim=0)
                labels_train = torch.cat([labels_orig, labels_aug], dim=0)
            else:
                audio_train = audio_orig
                labels_train = labels_orig

            # ---- Separate ALE Training ----
            if self.use_separate_ale_training and self.ale_optimizer is not None:
                if batch_idx % self.ale_update_freq == 0:
                    self.ale_optimizer.zero_grad()

                optimizer.zero_grad()
                outputs = self.model(audio_train, labels=labels_train)

                if outputs.dim() == 3:
                    labels_expanded = labels_train.unsqueeze(1).expand(-1, outputs.shape[1], -1)
                    loss = self.criterion(outputs, labels_expanded)
                else:
                    loss = self.criterion(outputs, labels_train)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.get_non_ale_params(), 1.0)
                optimizer.step()
                scheduler.step()
            else:
                # ---- Standard End-to-End Training ----
                optimizer.zero_grad()
                outputs = self.model(audio_train, labels=labels_train)

                if outputs.dim() == 3:
                    labels_expanded = labels_train.unsqueeze(1).expand(-1, outputs.shape[1], -1)
                    loss = self.criterion(outputs, labels_expanded)
                else:
                    loss = self.criterion(outputs, labels_train)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()

            # Pool frame-wise predictions to clip-level for metrics
            if outputs.dim() == 3:
                outputs_pooled = outputs.max(dim=1)[0]
            else:
                outputs_pooled = outputs

            all_predictions.extend(outputs_pooled.detach().cpu().numpy())
            all_targets.extend(labels_train.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            batch_idx += 1

        avg_loss = total_loss / len(loader)
        predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
        targets_binary = (np.array(all_targets) > 0.5).astype(int)
        f1 = f1_score(targets_binary, predictions_binary, average="macro", zero_division=0)

        return avg_loss, f1

    def validate(self, loader):
        """Validation loop returning (avg_loss, macro_f1)."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation", leave=False):
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(audio, labels=labels)

                if outputs.dim() == 3:
                    labels_expanded = labels.unsqueeze(1).expand(-1, outputs.shape[1], -1)
                    loss = self.criterion(outputs, labels_expanded)
                    outputs_pooled = outputs.max(dim=1)[0]
                else:
                    loss = self.criterion(outputs, labels)
                    outputs_pooled = outputs

                total_loss += loss.item()
                all_predictions.extend((outputs_pooled > 0.5).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(1, len(loader))
        predictions_binary = np.array(all_predictions).astype(int)
        targets_binary = np.array(all_targets).astype(int)
        f1 = f1_score(targets_binary, predictions_binary, average="macro", zero_division=0)

        return avg_loss, f1

    def _test_with_metrics(self, loader, dataset):
        """Test loop with advanced SED metrics (PSDS, event/segment based)."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_predictions_prob = []
        all_targets = []

        inference_start = time.time()
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing"):
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                num_samples += audio.size(0)

                outputs = self.model(audio, labels=labels)

                if outputs.dim() == 3:
                    labels_expanded = labels.unsqueeze(1).expand(-1, outputs.shape[1], -1)
                    loss = self.criterion(outputs, labels_expanded)
                    outputs_pooled = outputs.max(dim=1)[0]
                else:
                    loss = self.criterion(outputs, labels)
                    outputs_pooled = outputs

                total_loss += loss.item()
                all_predictions.extend((outputs_pooled > 0.5).cpu().numpy())
                all_predictions_prob.extend(outputs_pooled.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        inference_time = time.time() - inference_start
        avg_loss = total_loss / max(1, len(loader))
        predictions_binary = np.array(all_predictions).astype(int)
        predictions_prob = np.array(all_predictions_prob)
        targets_binary = np.array(all_targets).astype(int)

        basic_f1 = f1_score(targets_binary, predictions_binary, average="macro", zero_division=0)

        metrics_calculator = SEDMetricsCalculator(dataset.class_names)
        try:
            advanced_metrics = metrics_calculator.calculate_all_metrics(
                predictions_prob, targets_binary, dataset
            )
        except Exception as e:
            print(f"Warning: Advanced metrics failed: {e}")
            advanced_metrics = {
                "event_based": {"f1_macro": 0.0, "error_rate": 0.0},
                "segment_based": {"f1_macro": 0.0, "error_rate": 0.0},
                "psds1": 0.0, "psds2": 0.0,
            }

        print("\nTest Results:")
        print(f"  Loss: {avg_loss:.4f} | Basic F1: {basic_f1:.4f}")
        print(f"  Event-based   F1: {advanced_metrics['event_based']['f1_macro']:.4f}")
        print(f"  Event-based   ER: {advanced_metrics['event_based']['error_rate']:.4f}")
        print(f"  Segment-based F1: {advanced_metrics['segment_based']['f1_macro']:.4f}")
        print(f"  Segment-based ER: {advanced_metrics['segment_based']['error_rate']:.4f}")
        print(f"  PSDS: {advanced_metrics['psds1']:.4f}")
        print(f"  Inference: {inference_time:.2f}s total, "
              f"{inference_time / num_samples * 1000:.2f}ms/sample ({num_samples} samples)")

        return {
            "loss": avg_loss,
            "basic_f1": basic_f1,
            **advanced_metrics,
        }
