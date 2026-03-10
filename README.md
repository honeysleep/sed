# ACF-SED: ALE-Confidence Fusion Sound Event Detection

ACF-SED is a deep learning model for sleep-related sound event detection (snoring, hypopnea, obstructive apnea) using the APSAA (Apnea-related Polysomnography Sound and Annotation) dataset.

## Requirements

```
pip install -r requirements.txt
```

Core dependencies:
- `torch >= 2.0.0`
- `librosa >= 0.10.0`
- `sed_eval >= 0.2.1`
- `psds_eval >= 1.0.0`

## Dataset Structure

The APSAA dataset is organized per subject:
```
data_dir/
├── subject_001/
│   ├── subject_001.wav
│   └── subject_001_Annotations.csv
├── subject_002/
│   ├── subject_002.wav
│   └── subject_002_Annotations.csv
...
```

Each annotations CSV contains columns: `Event_Name`, `Start_Time`, `Duration`.
Target events: `snore`, `hypopnea`, `obstructive apnea`.

## Usage

### Training

```bash
python main.py \
    --data_dir /path/to/apsaa \
    --output_dir ./results \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --d_model 128 \
    --num_layers 4 \
    --mu_mode scalar
```

### Evaluation Only

```bash
python main.py \
    --data_dir /path/to/apsaa \
    --eval_only ./results/best_model_acf_sed.pth
```

### Resume Training

```bash
python main.py \
    --data_dir /path/to/apsaa \
    --output_dir ./results \
    --resume_from ./results/checkpoint_last.pth
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | (required) | Root directory of the dataset |
| `--d_model` | 128 | hidden dimension |
| `--nhead` | 4 | Number of attention heads |
| `--num_layers` | 4 | Number of DualPathConfindenceEncoderBlocks |
| `--mu_mode` | scalar | NLMS step-size mode: `fixed`/`scalar` |
| `--mu_init` | 0.05 | Initial NLMS step size |
| `--dropout` | 0.3 | Dropout rate |
| `--drop_path_rate` | 0.1 | Stochastic depth rate |
| `--use_augmentation` | True | Enable Mixup + FilterAug |
| `--window_mode` | fixed | `fixed` or `sliding` window extraction |
| `--use_separate_ale_training` | False | Separate optimizer for ALE params |

## Module Structure

```
test/
├── main.py              # Entry point
├── config.py            # Default hyperparameters
├── datasets.py          # SoundEventDataset
├── trainer.py           # SEDTrainer
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── ale_frontend.py  # NLMSFilter, MultiResolutionALEBank
│   ├── attention.py     # SymmetricConfidenceBiasedMHA, LearnableConfidencePooler
│   ├── modulation.py    # DropPath, ALEAwareModulation
│   ├── cnn_encoder.py   # EventCNN, NoiseCNN
│   ├── confindence_encoder_blocks.py  # _FeedForward, _LightweightFFN, _ConvModule,
│   │                         # DualPathConfidenceEncoderBlock
│   └── acf_sed.py       # ACF_SED, ACFJointLoss, create_acf_sed
└── utils/
    ├── __init__.py
    ├── metrics.py       # SEDMetricsCalculator (PSDS1/PSDS2, event/segment F1)
    └── augmentation.py  # apply_filter_aug, apply_balanced_mixup
```

## Metrics

The model is evaluated with:
- **Event-based F1** (sed_eval, 200ms collar)
- **Segment-based F1** (sed_eval, 1s segments)
- **PSDS**

