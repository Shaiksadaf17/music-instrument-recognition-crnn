"""
Data loading and preprocessing utilities for Music Genre Classification
and Instrument Recognition tasks.

Datasets:
- GTZAN: 1000 audio clips, 10 genres (30s each)
- IRMAS: ~6700 clips, 11 instruments (predominant instrument label)
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json


# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050
DURATION = 3          # seconds per clip (trim/pad to this)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FMAX = 8000

GTZAN_GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

IRMAS_INSTRUMENTS = [
    'cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi'
]
IRMAS_INSTRUMENT_NAMES = {
    'cel': 'Cello', 'cla': 'Clarinet', 'flu': 'Flute',
    'gac': 'Acoustic Guitar', 'gel': 'Electric Guitar', 'org': 'Organ',
    'pia': 'Piano', 'sax': 'Saxophone', 'tru': 'Trumpet',
    'vio': 'Violin', 'voi': 'Voice'
}


# ─── Feature Extraction ───────────────────────────────────────────────────────

def load_audio(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load audio, trim/pad to fixed duration."""
    import soundfile as sf
    import librosa
    
    try:
        # Try soundfile first (works natively on Windows)
        y, sr_native = sf.read(filepath, dtype='float32', always_2d=False)
        # Convert stereo to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Resample if needed
        if sr_native != sr:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=sr)
    except Exception:
        # Fallback to librosa
        y, _ = librosa.load(filepath, sr=sr, duration=duration)
    
    # Trim/pad to fixed duration
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_melspectrogram(y, sr=SAMPLE_RATE):
    """Extract log-mel spectrogram from audio signal."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # shape: (N_MELS, time_frames)


def extract_features_combined(y, sr=SAMPLE_RATE):
    """
    Extract a rich feature set: mel spectrogram + MFCCs + chroma + spectral contrast.
    Returns a stacked array suitable for multi-feature CNN input.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=N_FFT, hop_length=HOP_LENGTH)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Resize all to same time dimension (use mel as reference)
    T = log_mel.shape[1]
    mfcc = _resize_time(mfcc, T)
    chroma = _resize_time(chroma, T)
    contrast = _resize_time(contrast, T)

    return log_mel, mfcc, chroma, contrast


def _resize_time(arr, T):
    if arr.shape[1] >= T:
        return arr[:, :T]
    pad = T - arr.shape[1]
    return np.pad(arr, ((0, 0), (0, pad)))


# ─── GTZAN Dataset ────────────────────────────────────────────────────────────

class GTZANDataset(Dataset):
    """
    GTZAN dataset loader.

    Expected directory structure:
        gtzan_root/
            blues/
                blues.00000.wav
                ...
            classical/
                ...
    """

    def __init__(self, root_dir, split='train', test_size=0.2, val_size=0.1,
                 seed=42, augment=False, feature='mel'):
        self.root_dir = root_dir
        self.augment = augment
        self.feature = feature  # 'mel' or 'combined'

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(GTZAN_GENRES)

        filepaths, labels = [], []
        for genre in GTZAN_GENRES:
            genre_dir = os.path.join(root_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
            for fname in os.listdir(genre_dir):
                if fname.endswith('.wav') or fname.endswith('.au'):
                    filepaths.append(os.path.join(genre_dir, fname))
                    labels.append(genre)

        labels_enc = self.label_encoder.transform(labels)

        # Train / val / test split
        X_train, X_test, y_train, y_test = train_test_split(
            filepaths, labels_enc, test_size=test_size, stratify=labels_enc, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size), stratify=y_train, random_state=seed
        )

        if split == 'train':
            self.filepaths, self.labels = X_train, y_train
        elif split == 'val':
            self.filepaths, self.labels = X_val, y_val
        else:
            self.filepaths, self.labels = X_test, y_test

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        y = load_audio(self.filepaths[idx])
        if self.augment:
            y = augment_audio(y)

        if self.feature == 'mel':
            feat = extract_melspectrogram(y)
            feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
        else:
            log_mel, mfcc, chroma, contrast = extract_features_combined(y)
            feat = torch.tensor(np.vstack([log_mel, mfcc, chroma, contrast]), dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label


# ─── IRMAS Dataset ────────────────────────────────────────────────────────────

class IRMASDataset(Dataset):
    """
    IRMAS dataset loader (training set — single predominant instrument label).

    Expected directory structure:
        irmas_root/
            cel/
                [cel][nod][nnn]0001__1.wav
                ...
            cla/
                ...
    """

    def __init__(self, root_dir, split='train', test_size=0.2, val_size=0.1,
                 seed=42, augment=False):
        self.root_dir = root_dir
        self.augment = augment

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(IRMAS_INSTRUMENTS)

        filepaths, labels = [], []
        for instr in IRMAS_INSTRUMENTS:
            instr_dir = os.path.join(root_dir, instr)
            if not os.path.isdir(instr_dir):
                continue
            for fname in os.listdir(instr_dir):
                if fname.endswith('.wav'):
                    filepaths.append(os.path.join(instr_dir, fname))
                    labels.append(instr)

        labels_enc = self.label_encoder.transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            filepaths, labels_enc, test_size=test_size, stratify=labels_enc, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size), stratify=y_train, random_state=seed
        )

        if split == 'train':
            self.filepaths, self.labels = X_train, y_train
        elif split == 'val':
            self.filepaths, self.labels = X_val, y_val
        else:
            self.filepaths, self.labels = X_test, y_test

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        y = load_audio(self.filepaths[idx])
        if self.augment:
            y = augment_audio(y)
        feat = extract_melspectrogram(y)
        feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label


# ─── Data Augmentation ────────────────────────────────────────────────────────

def augment_audio(y, sr=SAMPLE_RATE):
    """Apply random augmentations: time stretch, pitch shift, noise."""
    choice = np.random.randint(3)
    if choice == 0:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
        target = sr * DURATION
        y = y[:target] if len(y) >= target else np.pad(y, (0, target - len(y)))
    elif choice == 1:
        n_steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    else:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise
    return y


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_gtzan_loaders(root_dir, batch_size=32, feature='mel', num_workers=0):
    train_ds = GTZANDataset(root_dir, split='train', augment=True, feature=feature)
    val_ds   = GTZANDataset(root_dir, split='val',   augment=False, feature=feature)
    test_ds  = GTZANDataset(root_dir, split='test',  augment=False, feature=feature)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def get_irmas_loaders(root_dir, batch_size=32, num_workers=0):
    train_ds = IRMASDataset(root_dir, split='train', augment=True)
    val_ds   = IRMASDataset(root_dir, split='val',   augment=False)
    test_ds  = IRMASDataset(root_dir, split='test',  augment=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
