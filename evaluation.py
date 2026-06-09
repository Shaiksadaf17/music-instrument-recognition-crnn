"""
Evaluation & Visualisation utilities
=====================================
Confusion matrices, training curves, qualitative audio analysis.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import librosa
import librosa.display


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, class_names, title='Confusion Matrix',
                          save_path=None, figsize=(10, 8)):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Saved → {save_path}')
    plt.close(fig)
    return fig


# ─── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(history, title='Training Curves', save_path=None):
    keys = [k for k in history if history[k]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'],   label='Val Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # Accuracy
    acc_keys = [k for k in history if 'acc' in k]
    for k in acc_keys:
        axes[1].plot(history[k], label=k)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Saved → {save_path}')
    plt.close(fig)
    return fig


# ─── Qualitative audio analysis ───────────────────────────────────────────────

def analyse_audio_file(filepath, genre_model, instr_model, fusion_model,
                       genre_classes, instr_classes, device,
                       save_dir='.', sr=22050, duration=3):
    """
    Run all three models on a single audio file and produce a qualitative
    analysis figure: waveform, mel spectrogram, top-5 predictions per model.
    """
    from utils.data_loader import load_audio, extract_melspectrogram

    y = load_audio(filepath, sr=sr, duration=duration)
    mel = extract_melspectrogram(y, sr=sr)
    mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    def get_probs(model):
        model.eval()
        with torch.no_grad():
            logits = model(mel_t)
            if isinstance(logits, tuple):
                return logits  # fusion model returns (genre, instr)
            return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    
    genre_probs = torch.softmax(genre_model(mel_t), dim=-1).detach().cpu().numpy()[0]
    instr_probs = torch.softmax(instr_model(mel_t), dim=-1).detach().cpu().numpy()[0]
    fused_genre, fused_instr = fusion_model(mel_t)
    fused_genre = torch.softmax(fused_genre, dim=-1).detach().cpu().numpy()[0]
    fused_instr = torch.softmax(fused_instr, dim=-1).detach().cpu().numpy()[0]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Waveform
    ax_wave = fig.add_subplot(gs[0, :])
    times = np.linspace(0, duration, len(y))
    ax_wave.plot(times, y, linewidth=0.5, color='steelblue')
    ax_wave.set_title(f'Waveform — {os.path.basename(filepath)}')
    ax_wave.set_xlabel('Time (s)')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.grid(alpha=0.3)

    # Mel spectrogram
    ax_mel = fig.add_subplot(gs[1, :2])
    img = librosa.display.specshow(mel, sr=sr, hop_length=512,
                                   x_axis='time', y_axis='mel',
                                   fmax=8000, ax=ax_mel)
    plt.colorbar(img, ax=ax_mel, format='%+2.0f dB')
    ax_mel.set_title('Log-Mel Spectrogram')

    def bar_chart(ax, probs, classes, title, color):
        top5_idx = np.argsort(probs)[::-1][:5]
        ax.barh([classes[i] for i in top5_idx[::-1]],
                [probs[i] for i in top5_idx[::-1]], color=color, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel('Probability')
        ax.grid(alpha=0.3, axis='x')

    bar_chart(fig.add_subplot(gs[1, 2]),   genre_probs, genre_classes,
              'GenreCNN (solo)',  'steelblue')
    bar_chart(fig.add_subplot(gs[2, 0]),   instr_probs, instr_classes,
              'InstrCRNN (solo)', 'coral')
    bar_chart(fig.add_subplot(gs[2, 1]),   fused_genre, genre_classes,
              'Fusion — Genre',   'seagreen')
    bar_chart(fig.add_subplot(gs[2, 2]),   fused_instr, instr_classes,
              'Fusion — Instr',   'orchid')

    fname = os.path.splitext(os.path.basename(filepath))[0]
    save_path = os.path.join(save_dir, f'qualitative_{fname}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Qualitative analysis saved → {save_path}')
    return save_path


# ─── Model comparison table ───────────────────────────────────────────────────

def print_model_comparison(results: dict):
    """
    results = {
        'model_name': {'genre_acc': ..., 'instr_acc': ..., 'genre_f1': ..., 'instr_f1': ...},
        ...
    }
    """
    header = f"{'Model':<25} {'Genre Acc':>10} {'Genre F1':>9} {'Instr Acc':>10} {'Instr F1':>9}"
    print('\n' + '=' * len(header))
    print(header)
    print('=' * len(header))
    for name, m in results.items():
        print(f"{name:<25} {m.get('genre_acc', 0):>10.4f} {m.get('genre_f1', 0):>9.4f} "
              f"{m.get('instr_acc', 0):>10.4f} {m.get('instr_f1', 0):>9.4f}")
    print('=' * len(header))
