import os
import numpy as np
import torch
import pandas as pd
import librosa
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset

CLASS_NAMES = ['snore', 'hypopnea', 'obstructive apnea']


class SoundEventDataset(Dataset):
    def __init__(self, data_dir, sample_rate=4000, window_size=10.0,
                 window_stride=5.0, subject_list=None, mode='sliding'):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_samples = int(window_size * sample_rate)
        self.stride_samples = int(window_stride * sample_rate)
        self.mode = mode
        self.samples = []
        self.class_names = CLASS_NAMES

        self._collect_data(subject_list)

        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Window size: {window_size}s ({self.window_samples} samples)")
        if mode == 'sliding':
            print(f"  Window stride: {window_stride}s ({self.stride_samples} samples)")
        self._print_class_distribution()

    def _collect_data(self, subject_list):
        if subject_list is None:
            subject_dirs = [d for d in os.listdir(self.data_dir)
                          if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
        else:
            all_dirs = [d for d in os.listdir(self.data_dir)
                       if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
            subject_dirs = [d for d in all_dirs if d in subject_list]

        for subject_id in tqdm(subject_dirs, desc="Loading data"):
            csv_path = os.path.join(self.data_dir, subject_id, f"{subject_id}_Annotations.csv")
            wav_path = os.path.join(self.data_dir, subject_id, f"{subject_id}.wav")

            if not (os.path.exists(csv_path) and os.path.exists(wav_path)):
                continue

            self._process_subject(subject_id, csv_path, wav_path)

    def _process_subject(self, subject_id, csv_path, wav_path):
        try:
            annotations = pd.read_csv(csv_path)
            if annotations.empty:
                return

            annotations['Event_Name'] = annotations['Event_Name'].astype(str).str.strip().str.lower()
            target_events = ['hypopnea', 'obstructive apnea', 'snore']
            annotations = annotations[annotations['Event_Name'].isin(target_events)].copy()

            if annotations.empty:
                return

            annotations['Start_Time_Seconds'] = annotations['Start_Time'].apply(self._time_to_seconds)
            annotations['Duration'] = annotations.get('Duration', 10.0)
            annotations['End_Time_Seconds'] = annotations['Start_Time_Seconds'] + annotations['Duration']
            annotations.dropna(subset=['Start_Time_Seconds'], inplace=True)

            if annotations.empty:
                return

            audio, _ = librosa.load(wav_path, sr=self.sample_rate, mono=True)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

            for _, row in annotations.iterrows():
                event_name = row['Event_Name']
                start_time = row['Start_Time_Seconds']
                duration = row['Duration']
                end_time = start_time + duration

                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)

                if end_sample > len(audio):
                    continue

                event_audio = audio[start_sample:end_sample]

                if self.mode == 'sliding':
                    self._add_sliding_windows(event_audio, event_name, subject_id, start_time, end_time)
                else:
                    self._add_fixed_window(event_audio, event_name, subject_id, start_time, end_time)

        except Exception as e:
            print(f"Warning: Error processing {subject_id}: {e}")

    def _add_sliding_windows(self, event_audio, event_name, subject_id, start_time, end_time):
        event_length = len(event_audio)

        if event_length <= self.window_samples:
            padded_audio = np.pad(event_audio, (0, self.window_samples - event_length), mode='constant')
            self._add_sample(padded_audio, event_name, subject_id, start_time, end_time)
        else:
            num_windows = (event_length - self.window_samples) // self.stride_samples + 1

            for i in range(num_windows):
                start_idx = i * self.stride_samples
                end_idx = start_idx + self.window_samples

                if end_idx > event_length:
                    start_idx = event_length - self.window_samples
                    end_idx = event_length

                window_audio = event_audio[start_idx:end_idx]
                window_start_time = start_time + (start_idx / self.sample_rate)
                window_end_time = start_time + (end_idx / self.sample_rate)

                self._add_sample(window_audio, event_name, subject_id, window_start_time, window_end_time)

                if end_idx >= event_length:
                    break

    def _add_fixed_window(self, event_audio, event_name, subject_id, start_time, end_time):
        if len(event_audio) < self.window_samples:
            event_audio = np.pad(event_audio, (0, self.window_samples - len(event_audio)), mode='constant')
        else:
            event_audio = event_audio[:self.window_samples]

        self._add_sample(event_audio, event_name, subject_id, start_time, end_time)

    def _time_to_seconds(self, time_str):
        if pd.isna(time_str) or time_str == '':
            return np.nan
        time_str = str(time_str).strip()

        try:
            return float(time_str)
        except ValueError:
            pass

        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            if len(parts) == 1:
                return float(parts[0])
            return np.nan
        except (ValueError, IndexError):
            return np.nan

    def _add_sample(self, audio, event_name, subject_id, start_time, end_time):
        self.samples.append({
            'audio': audio,
            'label': event_name,
            'subject_id': subject_id,
            'start_time': start_time,
            'end_time': end_time
        })

    def _print_class_distribution(self):
        class_counts = Counter([s['label'] for s in self.samples])
        print(f"\n  Class distribution:")
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            percentage = (count / len(self.samples) * 100) if len(self.samples) > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = sample['audio']

        if len(audio) != self.window_samples:
            if len(audio) < self.window_samples:
                audio = np.pad(audio, (0, self.window_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.window_samples]

        audio_tensor = torch.FloatTensor(audio)

        labels = np.zeros(len(self.class_names))
        if sample['label'] in self.class_names:
            label_idx = self.class_names.index(sample['label'])
            labels[label_idx] = 1
        labels = torch.FloatTensor(labels)

        return {
            'audio': audio_tensor,
            'labels': labels,
            'subject_id': sample['subject_id'],
            'start_time': sample.get('start_time', 0),
            'end_time': sample.get('end_time', 0)
        }


def summarize_dataset_events(dataset):
    """Summarize event counts and durations per class."""
    event_counts = {c: 0 for c in dataset.class_names}
    event_durations = {c: 0.0 for c in dataset.class_names}

    for sample in dataset.samples:
        label = sample['label']
        start = sample['start_time']
        end = sample['end_time']

        if label in event_counts:
            event_counts[label] += 1
            if not (start is None or end is None):
                event_durations[label] += (end - start)

    print("\nEvent Statistics:")
    print("  Event counts per class:")
    for cls in dataset.class_names:
        print(f"    {cls}: {event_counts[cls]}")

    print("\n  Total event durations (seconds) per class:")
    for cls in dataset.class_names:
        print(f"    {cls}: {event_durations[cls]:.2f}s ({event_durations[cls]/60:.2f}min)")

    total_events = sum(event_counts.values())
    total_duration = sum(event_durations.values())
    print(f"\n  Total events: {total_events}")
    print(f"  Total event duration: {total_duration:.2f}s ({total_duration/60:.2f}min)")
